#!/usr/bin/env python3
"""Error against GPU time, the comparison the methods are judged by.

    python3 eval/error_vs_time.py --scene cornell --render \\
        --configs A/hash/global B/mlhash/footprint B/mlrect/footprint \\
        --k-pix 2,4,8 --seeds 1,2,3 --frames 1024

A configuration is <family>/<accel>/<radius mode>[+variant...] as the renderer
names it (A: gather, B: gather from a culled map, C: splatting, D: the
adaptive LoD hash-grid SPPM, E: RTPM; see evalio.py for the variants, e.g.
B/hash/footprint+hier, D/sppm_lepski/footprint+kappa2, E/rtpm/global+cull),
or PT for path tracing with 4 samples per pixel per frame.  Footprint
configurations run once per --k-pix value, or per the values after an @
(C/mlsplat/footprint@4), or at the one their label names (+kpix8); global
ones ignore it.  For family D, k_pix is the radius of its largest scale.

With --render, every run that is not there yet is rendered, with checkpoints
at every power of two of frames, into

    eval/runs/<scene>/<F>_<accel>_<mode>[_k<K>]_d<D>_p<P>_r<W>x<H>_f<N>_s<seed>.*

(the image, its .checkpoints.json and .timing.json, and the .cpNNNN.exr
checkpoints) — every setting is in the name, and a run is only ever reused
under exactly that name.  Then, for every checkpoint,

    relMSE = mean over pixels and channels of (x - ref)^2 / (ref^2 + 0.01)

against the path-traced reference (render_references.py; the newest one
with the same scene, depth and resolution unless --reference names one), at
the checkpoint's cumulative GPU time: the sum of the per-pass GPU timers of
the frames so far, without file I/O or --stats.  The seeds are averaged per
checkpoint.  FLIP would need a package this machine does not have, so it is
not computed.

Written to eval/runs/<scene>/:
    error_vs_time.csv          every checkpoint of every run, and the means
    error_vs_time_k<K>.svg     one figure per k_pix (the global configurations
                               in each), log error against log time
    error_vs_time_best.svg     every configuration at its best k_pix
    error_vs_time.json         the series, fitted slopes, best k_pix

The slope is the least-squares fit of log relMSE against log time over the
checkpoints from --fit-from frames on; under the Knaus-Zwicker schedule an
exact method approaches -2/3.  The reference's own relMSE is drawn as a
dashed floor: a curve that reaches it measures the reference.
"""

import argparse
import csv
import glob
import math
import multiprocessing
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalio  # noqa: E402

RUNS = os.path.join(evalio.ROOT, "eval", "runs")
REFS = os.path.join(evalio.ROOT, "eval", "refs")

# The categorical palette of the figures, fixed slot order (light, dark).
SLOTS = (("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"),
         ("#eda100", "#c98500"), ("#e87ba4", "#d55181"), ("#008300", "#008300"),
         ("#6250d6", "#9085e9"), ("#e34948", "#e66767"))


def config_args(config):
    """Renderer options of a configuration name (evalio.config_options)."""
    if config.upper() == "PT":
        return []
    return ["--photon"] + evalio.config_options(config)


def k_pix_values(config, ks):
    """The k_pix values a configuration runs at: the given ones for a
    footprint configuration, none (the renderer's own, or the label's +kpixN)
    otherwise."""
    if config.upper() == "PT":
        return [None]
    footprint = evalio.split_config(config)[2] == "footprint"
    return ks if footprint and not evalio.has_variant(config, "kpix") else [None]


def run_name(config, k, args, seed):
    tag = "pt" if config.upper() == "PT" else config.replace("/", "_").replace("+", "_")
    if k is not None:
        tag += "_k%g" % k
    # --extra changes the run, so it is part of the name too
    extra = "".join(c for c in args.extra.replace(" ", "") if c.isalnum() or c in "-.")
    if extra and config.upper() != "PT":
        tag += "_x" + extra
    paths = "" if config.upper() == "PT" else "_p%d" % args.paths
    return "%s_d%d%s_r%dx%d_f%d_s%d" % (tag, args.depth, paths, args.res[0], args.res[1],
                                        args.frames, seed)


def find_reference(args):
    if args.reference:
        ref = args.reference
    else:
        pattern = os.path.join(REFS, "%s_pt_d%d_r%dx%d_f*_ref.exr" % (args.scene, args.depth,
                                                                     args.res[0], args.res[1]))
        found = sorted(glob.glob(pattern), key=lambda p: int(p.rsplit("_f", 1)[1].split("_")[0]))
        if not found:
            sys.exit("no reference for %s at depth %d, %dx%d: run eval/render_references.py"
                     % (args.scene, args.depth, args.res[0], args.res[1]))
        ref = found[-1]
    meta = ref[:-4] + ".json"
    info = evalio.load_json(meta) if os.path.exists(meta) else {}
    if info and (info.get("depth") != args.depth or info.get("res") != list(args.res)
                 or info.get("scene") != args.scene):
        sys.exit("%s was rendered for %s at depth %s, %s; these runs are %s at depth %d, %dx%d"
                 % (ref, info.get("scene"), info.get("depth"), info.get("res"), args.scene,
                    args.depth, args.res[0], args.res[1]))
    return ref, info


_REF = {}


def _init(ref_path):
    _, _, planes = evalio.read_exr(ref_path)
    _REF["planes"] = planes
    _REF["weights"] = evalio.ref_weights(planes)


def _error(path):
    _, _, planes = evalio.read_exr(path)
    return evalio.rel_mse(planes, _REF["planes"], _REF["weights"])


def fit_slope(points, min_frames):
    pts = [(math.log(t), math.log(e)) for f, t, e in points if f >= min_frames and e > 0 and t > 0]
    if len(pts) < 2:
        return float("nan")
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    return sum((x - mx) * (y - my) for x, y in pts) / sxx if sxx > 0 else float("nan")


# --------------------------------------------------------------------------
# figures

def svg_plot(path, title, subtitle, series, floor):
    """Log-log line figure: series = [(label, slot, [(frames, seconds, relmse)])]."""
    W, H = 820, 500
    L, R, T, B = 78, 200, 70, 56
    pw, ph = W - L - R, H - T - B
    xs = [math.log10(t) for _, _, pts in series for _, t, _ in pts if t > 0]
    ys = [math.log10(e) for _, _, pts in series for _, _, e in pts if e > 0]
    if floor:
        ys.append(math.log10(floor))
    if not xs or not ys:
        return
    x0, x1 = math.floor(min(xs)), math.ceil(max(xs))
    y0, y1 = math.floor(min(ys)), math.ceil(max(ys))
    x1 = x1 if x1 > x0 else x0 + 1
    y1 = y1 if y1 > y0 else y0 + 1

    def px(v):
        return L + (v - x0) / (x1 - x0) * pw

    def py(v):
        return T + (y1 - v) / (y1 - y0) * ph

    def sec(k):
        v = 10.0 ** k
        return ("%g s" % v) if v >= 0.01 else ("%g ms" % (v * 1000))

    out = []
    out.append('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" '
               'font-family="system-ui, -apple-system, Segoe UI, sans-serif">' % (W, H, W, H))
    css = [".bg{fill:var(--surface)}", ".grid{stroke:var(--grid);stroke-width:1}",
           ".axis{stroke:var(--axis);stroke-width:1}", ".t1{fill:var(--text-primary)}",
           ".t2{fill:var(--text-secondary)}", ".floor{stroke:var(--text-secondary);stroke-width:1.5;stroke-dasharray:6 4}"]
    light = ["--surface:#fcfcfb", "--grid:#e6e5e1", "--axis:#a8a7a2",
             "--text-primary:#0b0b0b", "--text-secondary:#52514e"]
    dark = ["--surface:#1a1a19", "--grid:#2e2e2c", "--axis:#5d5c58",
            "--text-primary:#f0efec", "--text-secondary:#c3c2b7"]
    for i, (lc, dc) in enumerate(SLOTS):
        light.append("--s%d:%s" % (i, lc))
        dark.append("--s%d:%s" % (i, dc))
        css.append(".l%d{stroke:var(--s%d);fill:none;stroke-width:2;stroke-linejoin:round}" % (i, i))
        css.append(".m%d{fill:var(--s%d);stroke:var(--surface);stroke-width:2}" % (i, i))
    out.append("<style>svg{%s}@media (prefers-color-scheme: dark){svg{%s}}%s</style>"
               % (";".join(light), ";".join(dark), "".join(css)))
    out.append('<rect class="bg" width="%d" height="%d"/>' % (W, H))
    out.append('<text class="t1" x="%d" y="28" font-size="16" font-weight="600">%s</text>' % (L, title))
    out.append('<text class="t2" x="%d" y="48" font-size="12">%s</text>' % (L, subtitle))

    # decade grid and ticks
    for k in range(x0, x1 + 1):
        x = px(k)
        out.append('<line class="grid" x1="%.1f" y1="%d" x2="%.1f" y2="%d"/>' % (x, T, x, T + ph))
        out.append('<text class="t2" x="%.1f" y="%d" font-size="11" text-anchor="middle">%s</text>'
                   % (x, T + ph + 18, sec(k)))
    for k in range(y0, y1 + 1):
        y = py(k)
        out.append('<line class="grid" x1="%d" y1="%.1f" x2="%d" y2="%.1f"/>' % (L, y, L + pw, y))
        out.append('<text class="t2" x="%d" y="%.1f" font-size="11" text-anchor="end">1e%d</text>'
                   % (L - 8, y + 4, k))
    out.append('<line class="axis" x1="%d" y1="%d" x2="%d" y2="%d"/>' % (L, T + ph, L + pw, T + ph))
    out.append('<line class="axis" x1="%d" y1="%d" x2="%d" y2="%d"/>' % (L, T, L, T + ph))
    out.append('<text class="t2" x="%.1f" y="%d" font-size="12" text-anchor="middle">GPU time</text>'
               % (L + pw / 2, H - 12))
    out.append('<text class="t2" x="18" y="%.1f" font-size="12" text-anchor="middle" '
               'transform="rotate(-90 18 %.1f)">relMSE</text>' % (T + ph / 2, T + ph / 2))

    if floor:
        y = py(math.log10(floor))
        out.append('<line class="floor" x1="%d" y1="%.1f" x2="%d" y2="%.1f"/>' % (L, y, L + pw, y))
        out.append('<text class="t2" x="%d" y="%.1f" font-size="11">reference noise</text>' % (L + 6, y - 6))

    # lines, markers, and labels at the line ends pushed apart vertically
    ends = []
    for label, slot, pts in series:
        good = [(math.log10(t), math.log10(e)) for _, t, e in pts if t > 0 and e > 0]
        if not good:
            continue
        coords = " ".join("%.1f,%.1f" % (px(a), py(b)) for a, b in good)
        out.append('<polyline class="l%d" points="%s"/>' % (slot, coords))
        for a, b in good:
            out.append('<circle class="m%d" cx="%.1f" cy="%.1f" r="4"/>' % (slot, px(a), py(b)))
        ends.append([py(good[-1][1]), px(good[-1][0]), label, slot])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 15)
    for y, x, label, slot in ends:
        out.append('<line class="l%d" x1="%d" y1="%.1f" x2="%d" y2="%.1f"/>' % (slot, L + pw + 8, y, L + pw + 22, y))
        out.append('<text class="t1" x="%d" y="%.1f" font-size="12">%s</text>' % (L + pw + 26, y + 4, label))
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True, choices=sorted(evalio.SCENES))
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--k-pix", default="4", help="comma-separated k_pix values for footprint configurations")
    p.add_argument("--seeds", default="1")
    p.add_argument("--frames", type=int, default=1024)
    p.add_argument("--paths", type=int, default=1000000)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--res", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    p.add_argument("--reference", help="the reference EXR (default: the newest matching one in eval/refs)")
    p.add_argument("--render", action="store_true", help="render the runs that are missing")
    p.add_argument("--fit-from", type=int, default=16, help="first checkpoint (frames) of the slope fit")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--out", default=RUNS)
    p.add_argument("--extra", default="", help="further renderer options for every photon run")
    args = p.parse_args()

    ks = [float(k) for k in args.k_pix.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    glb, cam = evalio.SCENES[args.scene]
    # Absolute: the renderer runs in its build directory.
    out_dir = os.path.join(os.path.abspath(args.out), args.scene)
    os.makedirs(out_dir, exist_ok=True)
    ref, ref_info = find_reference(args)
    floor = ref_info.get("relmse_reference")
    print("reference %s%s" % (os.path.relpath(ref, evalio.ROOT),
                              (", its own relMSE %.3g" % floor) if floor else ""))

    # The runs, rendering the missing ones.
    runs = []  # (config, k, seed, checkpoints json)
    configs, config_ks = [], {}
    for spec in args.configs:
        config, _, own = spec.partition("@")
        configs.append(config)
        config_ks[config] = [float(k) for k in own.split(",")] if own else ks
    args.configs = configs
    for config in args.configs:
        for k in k_pix_values(config, config_ks[config]):
            for seed in seeds:
                name = run_name(config, k, args, seed)
                stem = os.path.join(out_dir, name)
                cp_json = stem + ".checkpoints.json"
                if not os.path.exists(cp_json):
                    if not args.render:
                        print("missing %s (use --render)" % name)
                        continue
                    print("render %s" % name, flush=True)
                    # The configuration's own options last: a variant may
                    # override what is common to the runs (+paths300000).
                    extra = []
                    if config.upper() != "PT":
                        extra += ["--paths", args.paths, "--timing-json", stem + ".timing.json",
                                  "--warmup", 0] + args.extra.split()
                    extra += config_args(config) + (["--k-pix", k] if k is not None else [])
                    evalio.run_renderer([glb, "--offline", "--camera", cam, "--res", args.res[0],
                                         args.res[1], "--frames", args.frames, "--depth", args.depth,
                                         "--seed", seed, "--checkpoints", "--output", stem + ".exr"]
                                        + extra, stem + ".log")
                runs.append((config, k, seed, cp_json))

    # What the times were measured with.
    build_types = set()
    for config, k, seed, cp_json in runs:
        tj = cp_json.replace(".checkpoints.json", ".timing.json")
        if os.path.exists(tj):
            build_types.add(evalio.load_json(tj).get("environment", {}).get("build_type", "unknown"))
    if build_types - {"Release"}:
        print("WARNING: timed with a %s build — the times are not the thesis's Release numbers"
              % ", ".join(sorted(build_types - {"Release"})))

    # Every checkpoint's error, in parallel.
    tasks = []
    for config, k, seed, cp_json in runs:
        for cp in evalio.load_json(cp_json)["checkpoints"]:
            tasks.append((config, k, seed, cp["frames"], cp["gpu_ms"] / 1000.0, cp["file"]))
    print("%d checkpoints of %d runs against the reference, %d processes ..."
          % (len(tasks), len(runs), args.jobs), flush=True)
    with multiprocessing.Pool(args.jobs, initializer=_init, initargs=(ref,)) as pool:
        errors = pool.map(_error, [t[5] for t in tasks], chunksize=1)

    rows = []
    for (config, k, seed, frames, seconds, _), e in zip(tasks, errors):
        rows.append({"scene": args.scene, "config": config, "k_pix": k, "seed": seed,
                     "frames": frames, "gpu_s": seconds, "relmse": e})
    # Means over the seeds, per configuration, k_pix and checkpoint.
    groups = {}
    for r in rows:
        groups.setdefault((r["config"], r["k_pix"]), {}).setdefault(r["frames"], []).append(r)
    series = {}
    for key, by_frames in groups.items():
        pts = []
        for frames in sorted(by_frames):
            rs = by_frames[frames]
            pts.append((frames, sum(r["gpu_s"] for r in rs) / len(rs),
                        sum(r["relmse"] for r in rs) / len(rs)))
        series[key] = pts

    csv_path = os.path.join(out_dir, "error_vs_time.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "config", "k_pix", "seed", "frames", "gpu_s", "relmse"])
        for r in rows:
            w.writerow([r["scene"], r["config"], "" if r["k_pix"] is None else r["k_pix"], r["seed"],
                        r["frames"], "%.6f" % r["gpu_s"], "%.6g" % r["relmse"]])
        for (config, k), pts in sorted(series.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            for frames, t, e in pts:
                w.writerow([args.scene, config, "" if k is None else k, "mean", frames,
                            "%.6f" % t, "%.6g" % e])

    # Slopes, best k_pix per configuration (lowest error at the last checkpoint).
    summary = {"scene": args.scene, "reference": os.path.relpath(ref, evalio.ROOT),
               "reference_relmse": floor, "seeds": seeds, "frames": args.frames,
               "build_types": sorted(build_types), "renderer": evalio.RENDERER,
               "fit_from": args.fit_from, "series": [], "best_k": {}}
    print("\n%-28s %6s %10s %10s %9s" % ("configuration", "k_pix", "time (s)", "relMSE", "slope"))
    for (config, k), pts in sorted(series.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        slope = fit_slope(pts, args.fit_from)
        summary["series"].append({"config": config, "k_pix": k, "points": pts, "slope": slope})
        print("%-28s %6s %10.3f %10.4g %9.3f" % (config, "-" if k is None else "%g" % k,
                                                 pts[-1][1], pts[-1][2], slope))
    for config in args.configs:
        cands = [(pts[-1][2], k) for (c, k), pts in series.items() if c == config]
        if cands:
            summary["best_k"][config] = min(cands)[1]
    evalio.save_json(os.path.join(out_dir, "error_vs_time.json"), summary)

    # Figures: the slot follows the configuration, in the order given.
    slot = {c: i % len(SLOTS) for i, c in enumerate(args.configs)}
    note = "%s, %dx%d, depth %d, %s, mean of %d seed%s" % (
        args.scene, args.res[0], args.res[1], args.depth,
        "%d photon paths per frame" % args.paths, len(seeds), "" if len(seeds) == 1 else "s")
    for k in ks:
        sel = [(c + ("" if kk is None else " (k %g)" % kk), slot[c], series[(c, kk)])
               for c in args.configs for kk in (k, None) if (c, kk) in series]
        svg_plot(os.path.join(out_dir, "error_vs_time_k%g.svg" % k),
                 "Error against GPU time, k_pix = %g" % k, note, sel, floor)
    best = [(c + ("" if summary["best_k"].get(c) is None else " (k %g)" % summary["best_k"][c]),
             slot[c], series[(c, summary["best_k"].get(c))])
            for c in args.configs if (c, summary["best_k"].get(c)) in series]
    svg_plot(os.path.join(out_dir, "error_vs_time_best.svg"),
             "Error against GPU time, each at its best k_pix", note, best, floor)
    print("\n%s, error_vs_time_*.svg, error_vs_time.json" % os.path.relpath(csv_path, evalio.ROOT))


if __name__ == "__main__":
    main()
