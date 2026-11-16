#!/usr/bin/env python3
"""The fixed-radius variance diagnostic (E7): how much noise a one-photon-per-
bucket structure adds on top of the photon noise, and whether its speed pays
for it.

    python3 eval/variance.py --scene cornell --render \\
        --pairs B/mlhash:B/mlrect A/hash:A/rect --scales 1,0.5,0.25

For every pair exact:stochastic and every radius scale s, both run with the
schedule off (--no-ppm) and a per-pixel radius of 4 s pixel footprints
(--k-pix 4s), the same seed — so they trace identical photons from identical
gather points every frame — and save single frames (--dump-frames).  Then,
with m the exact run's mean over all its frames and

    relMSE(x, y) = mean over pixels and channels of (x - y)^2 / (m^2 + 0.01),

    V_exact = mean over the dumped frames of relMSE(exact_f, m)
              the photon noise of one frame;
    V_sel   = mean over the dumped frames of relMSE(stochastic_f, exact_f)
              what keeping one photon per bucket adds to it — the two see the
              same photons, so the difference is the selection alone;
    c       = (V_exact + V_sel) / V_exact
              the factor by which a stochastic frame is noisier;
    efficiency = (V_exact * t_exact) / ((V_exact + V_sel) * t_stochastic)
              > 1: the stochastic structure reaches a given variance sooner
              (t = mean GPU time per frame after the warm-up).

Runs go to eval/runs/<scene>/variance/, named by every setting; results to
eval/runs/<scene>/variance.json and .csv.
"""

import argparse
import csv
import multiprocessing
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalio  # noqa: E402

RUNS = os.path.join(evalio.ROOT, "eval", "runs")


def run_args(config, k, args):
    family, accel = config.split("/")
    opts = ["--photon", "--accel", accel, "--radius-mode", "footprint", "--k-pix", "%g" % k,
            "--no-ppm", "--paths", args.paths, "--warmup", args.warmup]
    if family != "C":  # splatting culls implicitly
        opts += ["--culling", "on" if family == "B" else "off"]
    return opts


_M = {}


def _init(mean_path):
    _, _, planes = evalio.read_exr(mean_path)
    _M["w"] = {c: [1.0 / (v * v + evalio.REL_MSE_EPS) for v in planes[c]] for c in "RGB"}
    _M["m"] = planes


def _pair_error(job):
    """relMSE(a, b) with the exact mean's weights; b = None means b = the mean."""
    a_path, b_path = job
    _, _, a = evalio.read_exr(a_path)
    b = _M["m"] if b_path is None else evalio.read_exr(b_path)[2]
    total, n = 0.0, 0
    for c in "RGB":
        w = _M["w"][c]
        total += sum((x - y) * (x - y) * k for x, y, k in zip(a[c], b[c], w))
        n += len(w)
    return total / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True, choices=sorted(evalio.SCENES))
    p.add_argument("--pairs", nargs="+", default=["B/mlhash:B/mlrect", "A/hash:A/rect"],
                   help="exact:stochastic, as <family>/<accel>")
    p.add_argument("--scales", default="1,0.5,0.25", help="radius scales s (k_pix = 4 s)")
    p.add_argument("--frames", type=int, default=256)
    p.add_argument("--dump-every", type=int, default=16, help="dump every n-th frame")
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--paths", type=int, default=1000000)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--res", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    p.add_argument("--render", action="store_true")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args()

    glb, cam = evalio.SCENES[args.scene]
    out = os.path.join(RUNS, args.scene, "variance")
    os.makedirs(out, exist_ok=True)
    dumps = list(range(args.dump_every, args.frames + 1, args.dump_every))
    rows = []
    for pair in args.pairs:
        exact, stoch = pair.split(":")
        for s in (float(x) for x in args.scales.split(",")):
            k = 4.0 * s
            runs = {}
            for config in (exact, stoch):
                name = "%s_k%g_noppm_d%d_p%d_r%dx%d_f%d_s%d" % (
                    config.replace("/", "_"), k, args.depth, args.paths, args.res[0], args.res[1],
                    args.frames, args.seed)
                stem = os.path.join(out, name)
                if not os.path.exists(stem + ".timing.json"):
                    if not args.render:
                        sys.exit("missing %s (use --render)" % name)
                    print("render %s" % name, flush=True)
                    evalio.run_renderer(
                        [glb, "--offline", "--camera", cam, "--res", args.res[0], args.res[1],
                         "--frames", args.frames, "--depth", args.depth, "--seed", args.seed,
                         "--dump-frames", "%d-%d:%d" % (args.dump_every, args.frames, args.dump_every),
                         "--timing-json", stem + ".timing.json", "--output", stem + ".exr"]
                        + run_args(config, k, args), stem + ".log")
                runs[config] = stem
            # The dumped frames are named by their absolute index (frame n of
            # the run is index n - 1).
            frame = lambda stem, n: "%s.frame%d.exr" % (stem, n - 1)  # noqa: E731
            with multiprocessing.Pool(args.jobs, initializer=_init, initargs=(runs[exact] + ".exr",)) as pool:
                v_exact = pool.map(_pair_error, [(frame(runs[exact], n), None) for n in dumps])
                v_sel = pool.map(_pair_error, [(frame(runs[stoch], n), frame(runs[exact], n)) for n in dumps])
            t = {}
            for config, stem in runs.items():
                frames = evalio.load_json(stem + ".timing.json")["per_frame"][args.warmup:]
                t[config] = sum(sum(fr.get(key, 0.0) for key in (
                    "camera_ms", "mark_ms", "photon_trace_ms", "build_ms", "query_ms",
                    "finalize_ms", "accum_ms")) for fr in frames) / len(frames)
            ve = sum(v_exact) / len(v_exact)
            vs = sum(v_sel) / len(v_sel)
            c = (ve + vs) / ve
            eff = (ve * t[exact]) / ((ve + vs) * t[stoch])
            rows.append({"scene": args.scene, "exact": exact, "stochastic": stoch, "scale": s,
                         "k_pix": k, "V_exact": ve, "V_sel": vs, "c": c,
                         "t_exact_ms": t[exact], "t_stochastic_ms": t[stoch], "efficiency": eff,
                         "frames": args.frames, "dumped": len(dumps)})
            print("%-9s vs %-9s s %-5g V_exact %.4g  V_sel %.4g  c %.3f  t %.2f / %.2f ms"
                  "  efficiency %.3f" % (exact, stoch, s, ve, vs, c, t[exact], t[stoch], eff),
                  flush=True)

    evalio.save_json(os.path.join(RUNS, args.scene, "variance.json"), rows)
    with open(os.path.join(RUNS, args.scene, "variance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
