#!/usr/bin/env python3
"""Render the same view in four modes in a single call.

    python3 exec.py                         # the 4 images into ./renders
    python3 exec.py --dry-run               # print the render commands only
    python3 exec.py --hash-frames 512       # more frames for one of them

The four images, every setting in the name:

    pt_f<F>_d<D>_s<S>.exr                  path tracing
    pm_hash_f<F>_d<D>_p<P>_s<S>.exr        photon mapping, full hash grid
    pm_stochastic_f<F>_d<D>_p<P>_s<S>.exr  photon mapping, stochastic hash grid
    pm_rectified_f<F>_d<D>_p<P>_s<S>.exr   photon mapping, rectified stochastic hash grid

All four share --depth and --seed.  The depth matters most: the shared
path-length budget is what makes photon mapping and path tracing solve the same
problem.  Each gets its own frame count, because a frame costs and buys very
different amounts in each mode:

    path tracing     1024  the reference, as converged as is affordable
                           (a frame here is 4 camera samples, not 1)
    full hash grid    256  every photon within the radius, the slow lookup
    stochastic       1024  one photon per bucket: a cheap lookup, but a much
    rectified        1024  noisier frame, so they need many more of them

More frames also means a smaller final radius under the PPM schedule
(r_N ~ N^-1/6 at alpha = 2/3), so a photon image at 1024 frames is somewhat
sharper, not only less noisy, than one at 256.

The three photon renders share --paths, so frame i traces the very same
photons from the very same gather points in all three, and any difference
between them is the lookup's.  The full hash grid adds up every photon within
the radius.  The stochastic grids keep one photon per bucket and scale it by
the bucket's photon count, so they estimate that same sum: the rectified grid
(reservoir sampling) without bias, the original one biased towards photons
deposited late in the trace (Davidovic et al. 2014, sec. 6.3).

Everything runs with cwd = build/, because the renderer resolves the scene and
the camera relative to ../scenes/.  Outputs are EXR, written to --out, with a
manifest.json recording the settings and the exact command of each image.

--radius-mode, --k-pix and --culling are passed on to the three photon
renders, and recorded in their file names when given:

    pm_<accel>[_footprint[_k<K>]][_culling]_f<F>_d<D>_p<P>_s<S>.exr

The benchmark (--bench) renders no image set.  It times the structures this
work compares, each over the same frames, and prints the table:

    python3 exec.py --bench                 # 10 configurations x 3 runs
    python3 exec.py --bench --dry-run       # the commands only

    hash/global                 full hash grid, one radius per frame
    hash/footprint              full hash grid, per-pixel radius (cell = the
                                largest radius of the frame)
    mlhash/footprint            multi-level hash grid, per-pixel radius
    mlhash/footprint+culling    the same, storing only reachable photons
    rect/global                 rectified stochastic hash grid, one radius
    mlrect/footprint            multi-level rectified grid, per-pixel radius
    mlrect/footprint+culling    the same, with culling
    sppm_lepski/footprint       adaptive LoD hash-grid SPPM (family D): its
                                own grid, culling and multi-scale estimate;
                                camera, photon, build and query are its four
                                stages
    rtpm/global                 RTPM (family E, Kern et al. 2023): the photons
                                in acceleration structures, collected by an
                                infinitely short ray; build is the structures,
                                query the collection
    rtpm/global+culling         the same with its Photon Culling

Every run is the renderer's own per-pass timing (--timing-json), averaged
over the frames after --bench-warmup; the table is the median over
--bench-runs runs of those averages, so a run disturbed by something else on
the machine does not move it.  Defaults: 1280x720, 1M photon paths, depth 8,
128 frames, 8 warm-up frames, 3 runs.  Timing JSONs, images and bench.json /
bench.md go to --bench-out (build/bench).
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
# build-release/, a Release build no IDE reconfigures (build/ follows whatever
# variant the editor's CMake integration picks), or RENDERER_BUILD.
BUILD = os.path.abspath(os.environ.get("RENDERER_BUILD") or (
    os.path.join(ROOT, "build-release") if os.path.isdir(os.path.join(ROOT, "build-release"))
    else os.path.join(ROOT, "build")))
RENDERER = os.path.join(BUILD, "renderer")
SRC = os.path.join(ROOT, "src")

# The renderer needs the discrete GPU even in offline mode.
RENDER_ENV = dict(os.environ)
RENDER_ENV["__NV_PRIME_RENDER_OFFLOAD"] = "1"
RENDER_ENV["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

# The four renders: the renderer's --accel value (None = path tracing), the
# prefix of its --<prefix>-frames option, what to call it, and its default
# frame count (see the table in the docstring).
MODES = (
    (None, "pt", "path tracing", 2048),
    ("hash", "hash", "photon mapping, full hash grid", 512),
    ("stochastic", "stochastic", "photon mapping, stochastic hash grid", 2048),
    ("rectified", "rectified", "photon mapping, rectified stochastic hash grid", 2048),
)


def frames_of(args, prefix):
    return getattr(args, prefix + "_frames")


def parse_args():
    p = argparse.ArgumentParser(
        description="Render the same view with path tracing and with photon "
                    "mapping through the full, stochastic and rectified "
                    "stochastic hash grids.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scene", default="scene.glb",
                   help="scene file, resolved by the renderer inside scenes/")
    p.add_argument("--camera", default="camera.txt",
                   help="camera file, resolved by the renderer inside scenes/")
    p.add_argument("--out", default=os.path.join(ROOT, "renders"),
                   help="directory to write the EXRs and manifest.json into")

    for _, prefix, label, frames in MODES:
        p.add_argument("--%s-frames" % prefix, type=int, default=frames,
                       metavar="N", help="frames to accumulate for " + label)
    p.add_argument("--depth", type=int, default=32,
                   help="max path length, the same for all four renders")
    p.add_argument("--seed", type=int, default=1,
                   help="random seed, the same for all four renders")
    p.add_argument("--paths", type=int, default=1_000_000,
                   help="photon paths per frame (photon mapping only)")

    p.add_argument("--radius-mode", choices=("global", "footprint"), default=None,
                   help="photon renders: one radius per frame, or one per pixel "
                        "from its footprint; omitted keeps the renderer default (global)")
    p.add_argument("--k-pix", type=float, default=None,
                   help="photon renders: footprint radius in pixels (renderer default 4)")
    p.add_argument("--culling", choices=("on", "off"), default=None,
                   help="photon renders: store only photons a lookup can reach")

    p.add_argument("--sky", type=float, default=None,
                   help="sky intensity; omitted keeps the renderer default (off)")
    p.add_argument("--sky-photons", type=int, choices=(0, 1), default=None,
                   help="emit sky photons; only meaningful together with --sky")

    p.add_argument("--build", action="store_true",
                   help="run cmake --build on build/ before rendering")
    p.add_argument("--force", action="store_true",
                   help="re-render images that already exist in --out")
    p.add_argument("--dry-run", action="store_true",
                   help="print the render commands, render nothing")

    b = p.add_argument_group("benchmark")
    b.add_argument("--bench", action="store_true",
                   help="time the seven configurations of the comparison instead "
                        "of rendering the image set")
    b.add_argument("--bench-runs", type=int, default=3, help="runs per configuration")
    b.add_argument("--bench-frames", type=int, default=128, help="frames per run")
    b.add_argument("--bench-warmup", type=int, default=8,
                   help="frames at the start of a run left out of its averages")
    b.add_argument("--bench-paths", type=int, default=1_000_000, help="photon paths per frame")
    b.add_argument("--bench-depth", type=int, default=8, help="max path length")
    b.add_argument("--bench-res", type=int, nargs=2, default=(1280, 720),
                   metavar=("W", "H"), help="resolution")
    b.add_argument("--bench-out", default=os.path.join(ROOT, "build", "bench"),
                   help="directory for the timing JSONs, images and the table")

    e = p.add_argument_group("experiments (section 9 of the method comparison)")
    e.add_argument("--experiments", action="store_true",
                   help="run the experiment matrix: E1-E4 (timing tables), E5-E6 (error "
                        "against time), E7 (variance diagnostic), E8 (the adaptive LoD "
                        "hash-grid SPPM) and E9 (every method, RTPM included): timing table "
                        "and error against time each, then --report")
    e.add_argument("--reduced", action="store_true",
                   help="the reduced matrix: 1 seed and 1024 frames for E5/E6/E8/E9, 256 frames for E7")
    e.add_argument("--only", default="E1,E2,E3,E4,E5,E6,E7,E8,E9",
                   help="comma-separated experiments to run")
    e.add_argument("--scenes", default="cornell,stadium,rooms,caustic",
                   help="comma-separated scenes (eval/evalio.py's names)")
    e.add_argument("--report", action="store_true",
                   help="write build/experiments/report.md from the results there")
    return p.parse_args()


def fail(msg):
    print("exec.py: " + msg, file=sys.stderr)
    sys.exit(1)


def check_environment(args):
    if not os.path.isdir(BUILD):
        fail("no build directory at %s — configure and build first (see readme.txt)" % BUILD)
    if not os.path.isfile(RENDERER):
        fail("no renderer binary at %s — build it first, or pass --build" % RENDERER)
    scene = os.path.join(ROOT, "scenes", args.scene)
    if not os.path.isfile(scene):
        fail("scene not found: %s" % scene)
    camera = os.path.join(ROOT, "scenes", args.camera)
    if not os.path.isfile(camera):
        fail("camera file not found: %s" % camera)

    # The renderer exits on an unknown option, so a stale binary would only fail
    # halfway through a long run.  Say so up front instead.
    binary_mtime = os.path.getmtime(RENDERER)
    newer = [f for f in sorted(os.listdir(SRC))
             if os.path.getmtime(os.path.join(SRC, f)) > binary_mtime]
    if newer:
        print("warning: %s is older than %s — rebuild if these renders should include it"
              % (os.path.relpath(RENDERER, ROOT), ", ".join(newer)))

    warn_build_type()


def warn_build_type():
    # Timings are only the thesis's numbers from a Release build (the timing
    # JSON of every render records which one it was).
    cache = os.path.join(BUILD, "CMakeCache.txt")
    build_type = None
    if os.path.isfile(cache):
        with open(cache) as f:
            for line in f:
                if line.startswith("CMAKE_BUILD_TYPE:"):
                    build_type = line.split("=", 1)[1].strip()
    if build_type != "Release":
        print("warning: %s is a %s build — times measured with it are not Release numbers"
              % (os.path.relpath(BUILD, ROOT), build_type or "unknown"))


def build():
    print("building ...")
    r = subprocess.run(["cmake", "--build", ".", "-j", str(os.cpu_count() or 4)],
                       cwd=BUILD)
    if r.returncode != 0:
        fail("build failed")


def image_name(args, accel, frames):
    if accel is None:
        return "pt_f%d_d%d_s%d.exr" % (frames, args.depth, args.seed)
    variant = ""
    if args.radius_mode == "footprint":
        variant += "_footprint"
        if args.k_pix is not None:
            variant += "_k%g" % args.k_pix
    if args.culling == "on":
        variant += "_culling"
    return "pm_%s%s_f%d_d%d_p%d_s%d.exr" % (accel, variant, frames, args.depth,
                                            args.paths, args.seed)


def render_cmd(args, *, frames, output, accel=None):
    """A path trace, or with `accel` a photon-mapped render through that structure."""
    cmd = [RENDERER, args.scene, "--offline",
           "--camera", args.camera,
           "--output", output,
           "--frames", str(frames),
           "--depth", str(args.depth),
           "--seed", str(args.seed)]
    if accel is not None:
        cmd.append("--photon")
        cmd += ["--paths", str(args.paths)]
        cmd += ["--accel", accel]
        # The renderer's own per-pass, per-frame timings, stored next to the
        # image — the trace/build/query split is where the three grids differ.
        cmd += ["--timing-json", os.path.splitext(output)[0] + ".timing.json"]
        if args.radius_mode is not None:
            cmd += ["--radius-mode", args.radius_mode]
        if args.k_pix is not None:
            cmd += ["--k-pix", str(args.k_pix)]
        if args.culling is not None:
            cmd += ["--culling", args.culling]
    if args.sky is not None:
        cmd += ["--sky", str(args.sky)]
    if args.sky_photons is not None:
        cmd += ["--sky-photons", str(args.sky_photons)]
    return cmd


def shell(cmd):
    """The command as the user would type it, for --dry-run and the manifest."""
    prefix = "__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia"
    parts = [os.path.relpath(c, BUILD) if c == RENDERER else c for c in cmd]
    parts[0] = "./" + parts[0]
    return prefix + " " + " ".join(parts)


def run_render(cmd, output, dry_run):
    """Run one render with its progress going straight to the terminal."""
    print("\n$ " + shell(cmd) + "   (cwd: %s/)" % os.path.relpath(BUILD, ROOT))
    if dry_run:
        return 0.0
    t0 = time.time()
    r = subprocess.run(cmd, cwd=BUILD, env=RENDER_ENV)
    seconds = time.time() - t0
    if r.returncode != 0:
        fail("render failed with exit code %d" % r.returncode)
    if not os.path.isfile(output) or os.path.getsize(output) == 0:
        fail("render reported success but wrote no image: %s" % output)
    print("  -> %s  (%s)" % (short_path(output), hms(seconds)))
    return seconds


def short_path(path):
    """Project-relative when the path is inside the project, absolute otherwise."""
    rel = os.path.relpath(path, ROOT)
    return path if rel.startswith("..") else rel


def hms(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "%dh%02dm%02ds" % (h, m, s)
    if m:
        return "%dm%02ds" % (m, s)
    return "%ds" % s


# The benchmark's configurations: a label and the renderer options behind it.
BENCH = (
    ("hash/global", ["--accel", "hash"]),
    ("hash/footprint", ["--accel", "hash", "--radius-mode", "footprint"]),
    ("mlhash/footprint", ["--accel", "mlhash", "--radius-mode", "footprint"]),
    ("mlhash/footprint+culling", ["--accel", "mlhash", "--radius-mode", "footprint",
                                  "--culling", "on"]),
    ("rect/global", ["--accel", "rectified"]),
    ("mlrect/footprint", ["--accel", "mlrect", "--radius-mode", "footprint"]),
    ("mlrect/footprint+culling", ["--accel", "mlrect", "--radius-mode", "footprint",
                                  "--culling", "on"]),
    ("sppm_lepski/footprint", ["--accel", "sppm_lepski"]),
    ("rtpm/global", ["--accel", "rtpm"]),
    ("rtpm/global+culling", ["--accel", "rtpm", "--rtpm-culling", "on"]),
)

# Per-pass times, in the order a frame runs them (mark: culling only;
# finalize: splatting only, whose build and photon pass are the point grid and
# the splatting pass; fused: --fused, the camera walk and the lookup in one).
PASSES = (("camera", "camera_ms"), ("mark", "mark_ms"), ("photon", "photon_trace_ms"),
          ("build", "build_ms"), ("query", "query_ms"), ("finalize", "finalize_ms"),
          ("fused", "fused_ms"), ("accum", "accum_ms"))

# Per-frame statistics averaged alongside the times.
STATS = ("bytes", "photons", "deposits", "culled", "entries", "buckets",
         "num_levels", "level_min", "level_max", "r_px_mean", "pairs", "reached")


def summarise_run(tjson, warmup):
    """Means over one run's frames after the warm-up."""
    with open(tjson) as f:
        frames = json.load(f)["per_frame"]
    frames = frames[warmup:] or frames
    out = {}
    for key, field in PASSES:
        out[key] = sum(fr.get(field, 0.0) for fr in frames) / len(frames)
    out["total"] = sum(out[key] for key, _ in PASSES)
    for key in STATS:
        out[key] = sum(fr.get(key, 0) for fr in frames) / len(frames)
    out["level_min"] = min(fr.get("level_min", -1) for fr in frames)
    out["level_max"] = max(fr.get("level_max", -1) for fr in frames)
    return out


def median(values):
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def bench(args):
    out = args.bench_out
    w, h = args.bench_res
    settings = "f%d_d%d_p%d_r%dx%d_s%d" % (args.bench_frames, args.bench_depth,
                                          args.bench_paths, w, h, args.seed)
    if not args.dry_run:
        os.makedirs(out, exist_ok=True)
    print("benchmark: %d configurations x %d runs, %dx%d, %d paths, depth %d, "
          "%d frames (%d warm-up), seed %d"
          % (len(BENCH), args.bench_runs, w, h, args.bench_paths, args.bench_depth,
             args.bench_frames, args.bench_warmup, args.seed))

    rows = []
    for label, extra in BENCH:
        tag = label.replace("/", "_").replace("+", "_")
        runs = []
        for k in range(args.bench_runs):
            stem = "%s_%s_run%d" % (tag, settings, k)
            output = os.path.join(out, stem + ".exr")
            tjson = os.path.join(out, stem + ".timing.json")
            cmd = [RENDERER, args.scene, "--offline", "--photon",
                   "--camera", args.camera,
                   "--res", str(w), str(h),
                   "--frames", str(args.bench_frames),
                   "--paths", str(args.bench_paths),
                   "--depth", str(args.bench_depth),
                   "--seed", str(args.seed),
                   "--warmup", str(args.bench_warmup),
                   "--timing-json", tjson,
                   "--output", output] + extra
            print("\n[%s, run %d/%d]" % (label, k + 1, args.bench_runs))
            run_render(cmd, output, args.dry_run)
            if not args.dry_run:
                runs.append(summarise_run(tjson, args.bench_warmup))
        if runs:
            med = {key: median([r[key] for r in runs]) for key in runs[0]}
            rows.append({"config": label, "options": extra, "median": med, "runs": runs})

    if args.dry_run:
        print("\ndry run: nothing was rendered")
        return

    head = ("| configuration | camera | photon | build | query | accum | total (ms) "
            "| structure (MiB) | stored | culled | entries | levels | T |")
    lines = [head, "|" + "---|" * 13]
    for r in rows:
        m = r["median"]
        levels = ("%.1f (%d..%d)" % (m["num_levels"], m["level_min"], m["level_max"])
                  if m["level_max"] >= 0 else "-")
        lines.append("| %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.1f | %.0f | %s | %s | %s | %s |" % (
            r["config"], m["camera"], m["photon"], m["build"], m["query"], m["accum"],
            m["total"], m["bytes"] / (1024.0 * 1024.0), m["photons"],
            "%.1f%%" % (100 * m["culled"]) if m["culled"] > 0 else "-",
            "%.0f" % m["entries"] if m["entries"] > 0 else "-", levels,
            "%.0f" % m["buckets"] if m["buckets"] > 0 else "-"))
    table = "\n".join(lines)
    note = ("ms per frame: per run the mean over frames %d..%d, then the median over "
            "%d runs.  structure = memory the lookup reads (not the photon map); "
            "stored = photons in the map; T = buckets."
            % (args.bench_warmup, args.bench_frames - 1, args.bench_runs))
    print("\n" + table + "\n\n" + note)
    with open(os.path.join(out, "bench_%s.md" % settings), "w") as f:
        f.write(table + "\n\n" + note + "\n")
    with open(os.path.join(out, "bench_%s.json" % settings), "w") as f:
        json.dump({"settings": {"res": [w, h], "frames": args.bench_frames,
                                "warmup": args.bench_warmup, "paths": args.bench_paths,
                                "depth": args.bench_depth, "seed": args.seed,
                                "runs": args.bench_runs, "scene": args.scene,
                                "camera": args.camera},
                   "rows": rows}, f, indent=2)
    print("table and per-run numbers in %s" % short_path(out))


# --------------------------------------------------------------------------
# The experiments (section 9).
#
# E1-E3 compare configurations that render the same image (or, for the
# one-photon-per-bucket grids, the same image in expectation): their per-pass
# times, totals and memory, at 1280x720, 1M photon paths, depth 8, the median
# of 3 runs of 128 frames after an 8-frame warm-up, per scene.
# E5-E6 compare error against GPU time (eval/error_vs_time.py), E7 is the
# variance diagnostic (eval/variance.py).

EXPERIMENTS = os.path.join(ROOT, "build", "experiments")
EVAL = os.path.join(ROOT, "eval")

SAME_IMAGE = {
    "E1": ("levels: one cell size against a level per lookup",
           ["A/hash/footprint", "A/mlhash/footprint", "A/rect/footprint", "A/mlrect/footprint"]),
    "E2": ("culling: the full photon map (A) against the culled one (B)",
           ["%s/%s/%s" % (f, a, m) for a in ("hash", "rect", "mlhash", "mlrect")
            for m in ("global", "footprint") for f in ("A", "B")]),
    "E3": ("gather against splat",
           ["B/mlhash/footprint", "C/mlsplat/footprint", "C/mlsplat/footprint+F1",
            "C/mlsplat/footprint+K4", "A/hash/global", "C/splat/global",
            "A/hash/footprint", "C/splat/footprint", "C/splat/footprint+F1"]),
    "E4": ("optional features: inline positions, hierarchical mask, fused family A",
           ["A/hash/footprint", "A/hash/footprint+inline", "A/mlhash/footprint",
            "A/mlhash/footprint+inline", "B/mlhash/footprint", "B/mlhash/footprint+inline",
            "C/mlsplat/footprint", "C/mlsplat/footprint+inline",
            "B/hash/footprint", "B/hash/footprint+hier", "B/mlhash/footprint+hier",
            "B/mlrect/footprint", "B/mlrect/footprint+hier",
            "A/hash/footprint+fused", "A/mlhash/footprint+fused", "A/hash/global",
            "A/hash/global+fused", "A/rect/footprint", "A/rect/footprint+fused",
            "A/mlrect/footprint", "A/mlrect/footprint+fused"]),
    # Families D and E do not go through the gathers' query pass, and D does
    # not render the image the others render — its estimate is its own — so
    # these two tables are about what a frame costs; what it buys is in the
    # error-against-time runs of the same experiments (ERROR_RUNS).
    "E8": ("adaptive LoD hash-grid SPPM (family D) against the hash grids",
           ["A/hash/footprint", "B/hash/footprint+hier", "B/mlrect/footprint",
            "D/sppm_lepski/footprint", "D/sppm_lepski/footprint+h0.08"]),
    "E9": ("every method: one frame's passes",
           ["A/hash/global", "A/hash/footprint", "A/hash/footprint+kpix8",
            "B/hash/footprint+hier", "B/mlhash/footprint", "A/rect/footprint",
            "B/mlrect/footprint", "C/mlsplat/footprint", "D/sppm_lepski/footprint",
            "E/rtpm/global", "E/rtpm/global+cull", "E/rtpm/global+cull+reject0.7+uniform",
            "E/rtpm/global+cull+reject0.7+uniform+stoch3", "E/rtpm/footprint+cull",
            "B/hash/global+hier+paths300000"]),
}

# Error against GPU time (eval/error_vs_time.py), per experiment: the
# configurations of one figure.  Footprint configurations run at k_pix 4
# unless they say otherwise (+kpix8).
#
# E8: family D at its defaults, with the selection's interval the method was
# specified with (+kappa2), and with its cell as well (+h0.08+kappa2).
# E9: every method the renderer has.  PT is the path tracer.  RTPM is there
# as an exact structure (E/rtpm/global, the hash grid's image), with its
# Photon Culling (+cull), and as the paper tested it: culling, 70% of the
# global photons rejected and the uniform kernel, without and with Stochastic
# Evaluation (+stoch3); E/rtpm/footprint is RTPM with per-pixel radii.  The
# last one is the control for that rejection: the culled hash grid given 30%
# of the photon paths, so about as many photons as RTPM keeps.
ERROR_RUNS = {
    "E8": ["A/hash/footprint", "B/mlrect/footprint", "D/sppm_lepski/footprint",
           "D/sppm_lepski/footprint+kappa2", "D/sppm_lepski/footprint+h0.08+kappa2"],
    "E9": ["PT"] + SAME_IMAGE["E9"][1],
}
ERROR_TITLES = {
    "E8": "adaptive LoD hash-grid SPPM, error against GPU time",
    "E9": "every method, error against GPU time",
}


def config_options(label):
    """Renderer options of '<family>/<accel>/<mode>[+variant...]' (eval/evalio.py)."""
    sys.path.insert(0, EVAL)
    import evalio
    return evalio.config_options(label)


def scene_files(scene):
    sys.path.insert(0, EVAL)
    import evalio
    return evalio.SCENES[scene]


def run_same_image(args, scenes, which):
    """E1-E3: every configuration they name, per scene, 3 runs; the medians."""
    labels = []
    for e in which:
        for label in SAME_IMAGE[e][1]:
            if label not in labels:
                labels.append(label)
    results = {}
    for scene in scenes:
        glb, cam = scene_files(scene)
        out = os.path.join(EXPERIMENTS, "same_image", scene)
        if not args.dry_run:
            os.makedirs(out, exist_ok=True)
        for label in labels:
            runs = []
            for k in range(args.bench_runs):
                stem = os.path.join(out, "%s_run%d" % (label.replace("/", "_").replace("+", "_"), k))
                tjson = stem + ".timing.json"
                cmd = [RENDERER, glb, "--offline", "--photon", "--camera", cam,
                       "--res", "1280", "720", "--frames", str(args.bench_frames),
                       "--paths", "1000000", "--depth", "8", "--seed", str(args.seed),
                       "--warmup", str(args.bench_warmup), "--timing-json", tjson,
                       "--output", stem + ".exr"] + config_options(label)
                if os.path.exists(tjson) and not args.force:
                    print("  %s %s run %d: done" % (scene, label, k + 1))
                else:
                    print("\n[%s, %s, run %d/%d]" % (scene, label, k + 1, args.bench_runs))
                    run_render(cmd, stem + ".exr", args.dry_run)
                if not args.dry_run:
                    runs.append(summarise_run(tjson, args.bench_warmup))
            if runs:
                med = {key: median([r[key] for r in runs]) for key in runs[0]}
                results.setdefault(scene, {})[label] = {"median": med, "runs": runs}
    if not args.dry_run:
        os.makedirs(EXPERIMENTS, exist_ok=True)
        path = os.path.join(EXPERIMENTS, "same_image.json")
        old = {}
        if os.path.exists(path):
            with open(path) as f:
                old = json.load(f)
        for scene, rows in results.items():
            old.setdefault(scene, {}).update(rows)
        with open(path, "w") as f:
            json.dump(old, f, indent=2)


def run_tool(cmd, dry_run):
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return
    r = subprocess.run(cmd)
    if r.returncode != 0:
        fail("%s failed" % os.path.basename(cmd[1]))


def experiments(args):
    scenes = args.scenes.split(",")
    which = [e.strip() for e in args.only.split(",")]
    frames = 1024 if args.reduced else 4096
    seeds = "1" if args.reduced else "1,2,3"
    same = [e for e in which if e in SAME_IMAGE]
    if same:
        run_same_image(args, scenes, same)
    for scene in scenes:
        if "E5" in which:
            run_tool([sys.executable, os.path.join(EVAL, "error_vs_time.py"), "--scene", scene,
                      "--render", "--frames", str(frames), "--seeds", seeds, "--k-pix", "2,4,8",
                      "--out", os.path.join(EXPERIMENTS, "E5"), "--configs",
                      "A/hash/footprint", "A/rect/footprint", "B/mlhash/footprint",
                      "B/mlrect/footprint"] + (["C/mlsplat/footprint@4", "C/stochsplat/footprint@4"]
                                               if args.reduced else
                                               ["C/mlsplat/footprint", "C/stochsplat/footprint"]),
                     args.dry_run)
        if "E6" in which:
            run_tool([sys.executable, os.path.join(EVAL, "error_vs_time.py"), "--scene", scene,
                      "--render", "--frames", str(frames), "--seeds", seeds, "--k-pix", "4",
                      "--out", os.path.join(EXPERIMENTS, "E6"), "--configs",
                      "B/mlhash/global", "B/mlhash/footprint", "A/hash/global",
                      "A/hash/footprint"], args.dry_run)
        if "E7" in which:
            run_tool([sys.executable, os.path.join(EVAL, "variance.py"), "--scene", scene,
                      "--render", "--frames", str(256 if args.reduced else 1024),
                      "--dump-every", "16", "--pairs", "B/mlhash:B/mlrect", "A/hash:A/rect",
                      "C/mlsplat:C/stochsplat"], args.dry_run)
        for e in which:
            if e in ERROR_RUNS:
                run_tool([sys.executable, os.path.join(EVAL, "error_vs_time.py"), "--scene", scene,
                          "--render", "--frames", str(frames), "--seeds", seeds, "--k-pix", "4",
                          "--out", os.path.join(EXPERIMENTS, e), "--configs"] + ERROR_RUNS[e],
                         args.dry_run)
    if not args.dry_run:
        write_report()


def write_report():
    """build/experiments/report.md from what the experiments left there."""
    lines = ["# Photon-gather method families: experiment results", ""]
    path = os.path.join(EXPERIMENTS, "same_image.json")
    if os.path.exists(path):
        with open(path) as f:
            same = json.load(f)
        for e, (title, labels) in SAME_IMAGE.items():
            lines += ["## %s — %s" % (e, title), ""]
            for scene, rows in same.items():
                present = [l for l in labels if l in rows]
                if not present:
                    continue
                lines += ["**%s** (ms per frame, median of %d runs)" % (scene, len(rows[present[0]]["runs"])), "",
                          "| configuration | camera | mark | photon | build | query | finalize | fused | total "
                          "| structure MiB | stored | culled | entries | T |",
                          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
                for l in present:
                    m = rows[l]["median"]
                    lines.append("| %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | **%.2f** | %.1f | %.0f | %s | %.0f | %.0f |" % (
                        l, m["camera"], m.get("mark", 0), m["photon"], m["build"], m["query"],
                        m.get("finalize", 0), m.get("fused", 0), m["total"], m["bytes"] / 1048576.0, m["photons"],
                        "%.1f%%" % (100 * m["culled"]) if m["culled"] > 0 else "-",
                        m["entries"], m["buckets"]))
                lines.append("")
    for e, title in (("E5", "stochastic against exact, error against GPU time"),
                     ("E6", "radius policy, error against GPU time")) + tuple(ERROR_TITLES.items()):
        base = os.path.join(EXPERIMENTS, e)
        if not os.path.isdir(base):
            continue
        lines += ["## %s — %s" % (e, title), ""]
        for scene in sorted(os.listdir(base)):
            js = os.path.join(base, scene, "error_vs_time.json")
            if not os.path.exists(js):
                continue
            with open(js) as f:
                s = json.load(f)
            lines += ["**%s** — reference relMSE %.2g; %d frames, seeds %s; figures in %s" % (
                scene, s.get("reference_relmse") or float("nan"), s["frames"],
                ",".join(str(x) for x in s["seeds"]), os.path.relpath(os.path.join(base, scene), ROOT)),
                "", "| configuration | k_pix | GPU s | relMSE at the end | slope from 16 frames |",
                "|---|---|---|---|---|"]
            for ser in s["series"]:
                pts = ser["points"]
                lines.append("| %s | %s | %.2f | %.4g | %.3f |" % (
                    ser["config"], "-" if ser["k_pix"] is None else "%g" % ser["k_pix"],
                    pts[-1][1], pts[-1][2], ser["slope"]))
            lines += ["", "best k_pix: " + ", ".join("%s %s" % (c, "-" if k is None else "%g" % k)
                                                     for c, k in s["best_k"].items()), ""]
    var_rows = []
    runs = os.path.join(ROOT, "eval", "runs")
    if os.path.isdir(runs):
        for scene in sorted(os.listdir(runs)):
            js = os.path.join(runs, scene, "variance.json")
            if os.path.exists(js):
                with open(js) as f:
                    var_rows += json.load(f)
    if var_rows:
        lines += ["## E7 — fixed-radius variance diagnostic", "",
                  "| scene | exact | stochastic | s | V_exact | V_sel | c | t exact | t stochastic | efficiency |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in var_rows:
            lines.append("| %s | %s | %s | %g | %.4g | %.4g | %.3f | %.2f | %.2f | %.3f |" % (
                r["scene"], r["exact"], r["stochastic"], r["scale"], r["V_exact"], r["V_sel"],
                r["c"], r["t_exact_ms"], r["t_stochastic_ms"], r["efficiency"]))
        lines.append("")
    os.makedirs(EXPERIMENTS, exist_ok=True)
    out = os.path.join(EXPERIMENTS, "report.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("report: %s" % short_path(out))


def main():
    args = parse_args()
    if args.build:
        build()
    if args.report and not args.experiments:
        write_report()
        return
    if args.experiments:
        if not os.path.isfile(RENDERER):
            fail("no renderer binary at %s" % RENDERER)
        warn_build_type()
        experiments(args)
        return
    check_environment(args)
    if args.bench:
        bench(args)
        return

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")

    manifest = {
        "scene": args.scene,
        "camera": args.camera,
        "depth": args.depth,
        "seed": args.seed,
        "paths": args.paths,
        "sky": args.sky,
        "sky_photons": args.sky_photons,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "renders": [],
    }

    def save_manifest():
        if args.dry_run:
            return
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print("%d renders: depth %d, seed %d, %d photon paths/frame"
          % (len(MODES), args.depth, args.seed, args.paths))

    for i, (accel, prefix, label, _) in enumerate(MODES):
        frames = frames_of(args, prefix)
        name = image_name(args, accel, frames)
        output = os.path.join(args.out, name)
        cmd = render_cmd(args, frames=frames, output=output, accel=accel)
        entry = {"file": name, "mode": label, "accel": accel, "frames": frames,
                 "command": shell(cmd)}
        if os.path.exists(output) and not args.force:
            print("\n[%d/%d] %s: already rendered, %s (--force to redo)"
                  % (i + 1, len(MODES), label, name))
            entry["skipped"] = True
        else:
            print("\n[%d/%d] %s, %d frames" % (i + 1, len(MODES), label, frames))
            entry["seconds"] = run_render(cmd, output, args.dry_run)
        manifest["renders"].append(entry)
        save_manifest()

    if args.dry_run:
        print("\ndry run: nothing was rendered")
        return

    print("\ndone — images in %s" % short_path(args.out))
    for entry in manifest["renders"]:
        print("  %-44s %s" % (entry["file"],
                              "reused" if entry.get("skipped") else hms(entry["seconds"])))
    print("  %-44s render settings and commands" % "manifest.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
