#!/usr/bin/env python3
"""Checks of the photon-gather families (A: gather, B: gather from a culled
map, C: splatting).  Every check runs the renderer itself and compares what it
wrote:

    python3 checks.py v1        infrastructure: every existing configuration
                                renders the same image as before (the frozen
                                build in build/pre_families), plus the timing
                                table and the kernels' registers
    python3 checks.py timing    the per-pass time comparison of v1 on its own
    python3 checks.py v2        range-based visits: same images, visited cells
                                and candidates per lookup before and after,
                                exact structures against brute force at
                                frames 1, 100 and 1000
    python3 checks.py v3        mask decoupling: culled = unculled per frame for
                                the exact structures, tested entries per lookup
                                of B/mlhash, and the one-photon-per-bucket grids
                                unbiased with and without culling
    python3 checks.py v4        splatting: splat and mlsplat against hash and
                                mlhash, per frame, at frames 1, 100 and 1000
    python3 checks.py v5        rect, mlrect and stochastic splatting close in on
                                hash, mlhash and mlsplat like 1/sqrt(M)
    python3 checks.py v6        --fused renders the same images as the split
                                camera and query passes
    python3 checks.py v7        error against time: the Knaus-Zwicker slope on
                                the Cornell box against a path-traced reference
    python3 checks.py diff A.exr B.exr    the image comparison on its own

Images, logs and timing JSON go to build/checks/<check>/, with every setting
that produced them in the file name.  A check never reads an image it did not
render in the same run, so a stale file cannot leak into a result.

The comparison reads the uncompressed scanline EXRs the renderer writes,
without numpy: struct unpacks a whole scanline per call, which keeps a
1280x720 image to about a second.  Renders for comparisons use --float-exr;
half floats keep 11 significant bits, far too few for a 1e-5 tolerance.
"""

import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
from array import array

ROOT = os.path.dirname(os.path.abspath(__file__))
# The renderer under test: build-release/, a Release build no IDE reconfigures
# (build/ follows whatever variant the editor's CMake integration picks), or
# the directory RENDERER_BUILD names.
BUILD = os.path.abspath(os.environ.get("RENDERER_BUILD") or (
    os.path.join(ROOT, "build-release") if os.path.isdir(os.path.join(ROOT, "build-release"))
    else os.path.join(ROOT, "build")))
OUT = os.path.join(ROOT, "build", "checks")
# The sources as they were before this work, built in Release (see
# build/pre_families/tree_state.txt): the "before" of every comparison.
PRE = os.path.join(ROOT, "build", "pre_families", "build")

# The renderer needs the discrete GPU even in offline mode (as in exec.py).
RENDER_ENV = dict(os.environ)
RENDER_ENV["__NV_PRIME_RENDER_OFFLOAD"] = "1"
RENDER_ENV["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

SCENE = "scene.glb"
CAMERA = "camera.txt"

# Relative differences are only meaningful where there is signal; below this
# the absolute difference is what is reported (the same floor verifyQueryPass
# uses).
REL_FLOOR = 1e-4
ALLCLOSE_RTOL, ALLCLOSE_ATOL = 1e-5, 1e-8

# Brute force tests every photon for every pixel, so at the default resolution
# and photon count a frame takes minutes; it runs on a smaller image and map.
SMALL = ["--res", 320, 180, "--paths", 50000]

# The benchmark settings of the thesis tables.
BENCH = ["--res", 1280, 720, "--paths", 1000000, "--depth", 8]


def fail(msg):
    print("checks.py: " + msg, file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------
# EXR reading and comparison

def read_exr(path):
    """(width, height, {channel: array('f')}) of an uncompressed scanline EXR."""
    with open(path, "rb") as f:
        data = f.read()
    magic, version = struct.unpack_from("<ii", data, 0)
    if magic != 20000630:
        raise ValueError("%s: not an OpenEXR file" % path)
    if version & 0x200:
        raise ValueError("%s: tiled EXR not supported" % path)

    off = 8
    channels, compression, window = [], None, None
    while True:
        end = data.index(b"\0", off)
        name = data[off:end].decode()
        off = end + 1
        if not name:
            break
        end = data.index(b"\0", off)
        off = end + 1  # attribute type, implied by the name for the ones used
        (size,) = struct.unpack_from("<i", data, off)
        off += 4
        value = data[off:off + size]
        off += size
        if name == "channels":
            p = 0
            while value[p] != 0:
                e = value.index(b"\0", p)
                cname = value[p:e].decode()
                (ptype,) = struct.unpack_from("<i", value, e + 1)
                channels.append((cname, ptype))
                p = e + 1 + 16
        elif name == "compression":
            compression = value[0]
        elif name == "dataWindow":
            window = struct.unpack("<iiii", value)
    if compression != 0:
        raise ValueError("%s: compressed EXR (type %s) not supported" % (path, compression))

    xmin, ymin, xmax, ymax = window
    w, h = xmax - xmin + 1, ymax - ymin + 1
    offsets = struct.unpack_from("<%dQ" % h, data, off)
    rows = {c: [None] * h for c, _ in channels}
    for o in offsets:
        (y,) = struct.unpack_from("<i", data, o)
        p = o + 8
        for cname, ptype in channels:
            if ptype == 1:
                rows[cname][y - ymin] = struct.unpack_from("<%de" % w, data, p)
                p += 2 * w
            elif ptype == 2:
                rows[cname][y - ymin] = struct.unpack_from("<%df" % w, data, p)
                p += 4 * w
            else:
                raise ValueError("%s: pixel type %d not supported" % (path, ptype))
    planes = {}
    for cname, _ in channels:
        a = array("f")
        for r in rows[cname]:
            a.extend(r)
        planes[cname] = a
    return w, h, planes


def compare(path_a, path_b):
    """Differences between two RGB images, b measured against a."""
    wa, ha, a = read_exr(path_a)
    wb, hb, b = read_exr(path_b)
    if (wa, ha) != (wb, hb):
        raise ValueError("size mismatch: %dx%d vs %dx%d" % (wa, ha, wb, hb))
    max_abs = max_rel = 0.0
    sum_abs = sum_ref = sum_sq = sum_ref_sq = 0.0
    over_1e5 = over_1e3 = not_close = 0
    for ch in "RGB":
        for x, y in zip(a[ch], b[ch]):
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("non-finite value in %s or %s" % (path_a, path_b))
            d = abs(x - y)
            sum_abs += d
            sum_sq += d * d
            sum_ref += abs(x)
            sum_ref_sq += x * x
            if d > max_abs:
                max_abs = d
            scale = max(abs(x), abs(y))
            # numpy's allclose with its default tolerances
            if d > ALLCLOSE_ATOL + ALLCLOSE_RTOL * scale:
                not_close += 1
            if scale > REL_FLOOR:
                rel = d / scale
                if rel > max_rel:
                    max_rel = rel
                if rel > 1e-5:
                    over_1e5 += 1
                if rel > 1e-3:
                    over_1e3 += 1
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "rel_l1": sum_abs / sum_ref if sum_ref > 0 else 0.0,
        "rel_l2": math.sqrt(sum_sq / sum_ref_sq) if sum_ref_sq > 0 else 0.0,
        "values_over_1e-5": over_1e5,
        "values_over_1e-3": over_1e3,
        "not_allclose": not_close,
    }


def same_image(d):
    """The verdict for images that should be the same up to float noise:
    within 1e-5 relative wherever there is signal, or else every value within
    numpy's allclose (rtol 1e-5, atol 1e-8).  The second matters for the
    one-photon-per-bucket grids: a pixel whose value is one representative
    with a cone weight near zero changes by 1e-5 of itself when the photon's
    position moves in the last bit, while being 1e-8 or less in absolute
    terms."""
    return d["max_rel"] <= 1e-5 or d["not_allclose"] == 0


# --------------------------------------------------------------------------
# Running the renderer

def renderer_dir(path):
    """A directory holding a built renderer and its device.ptx.  It is also
    the renderer's working directory, so ../scenes must exist next to it."""
    path = os.path.abspath(path)
    for f in ("renderer", "device.ptx"):
        if not os.path.isfile(os.path.join(path, f)):
            fail("no %s in %s" % (f, path))
    if not os.path.isfile(os.path.join(path, "..", "scenes", SCENE)):
        fail("%s has no ../scenes/%s" % (path, SCENE))
    return path


def render(bin_dir, out_dir, name, args):
    """Runs one offline photon-mapped render; returns (exr, timing json, stdout)."""
    os.makedirs(out_dir, exist_ok=True)
    exr = os.path.join(out_dir, name + ".exr")
    tjson = os.path.join(out_dir, name + ".timing.json")
    log = os.path.join(out_dir, name + ".log")
    cmd = [os.path.join(bin_dir, "renderer"), SCENE, "--offline", "--photon",
           "--camera", CAMERA, "--float-exr", "--output", exr,
           "--timing-json", tjson] + [str(a) for a in args]
    print("  render %-66s" % name, end="", flush=True)
    r = subprocess.run(cmd, cwd=bin_dir, env=RENDER_ENV,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n\n" + r.stdout)
    if r.returncode != 0:
        print(" FAILED")
        fail("render failed (exit %d), log in %s" % (r.returncode, log))
    if not os.path.isfile(exr):
        fail("render wrote no image, log in %s" % log)
    print(" ok")
    return exr, tjson, r.stdout


PASS_KEYS = ("camera_ms", "mark_ms", "photon_trace_ms", "build_ms", "query_ms", "accum_ms")


def timing_means(tjson, warmup):
    """Mean ms per pass over the frames after the warm-up, and the means of
    the per-frame statistics that are there."""
    with open(tjson) as f:
        t = json.load(f)
    frames = t["per_frame"][warmup:] or t["per_frame"]
    out = {}
    for k in PASS_KEYS + ("photons", "deposits", "culled", "entries", "buckets", "bytes",
                          "tested_mean", "accepted_mean", "accepted_p95", "accepted_max",
                          "cells_mean", "num_levels"):
        if k in frames[0]:
            out[k] = sum(fr[k] for fr in frames) / len(frames)
    out.setdefault("mark_ms", 0.0)
    out["total_ms"] = sum(out[k] for k in PASS_KEYS)
    return out


def fmt(x):
    return "%.3g" % x


def median(values):
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def tag_of(args):
    return "_".join(str(a).lstrip("-") for a in args).replace("/", "")


def write_results(out, name, obj):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name), "w") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------
# V1 — the infrastructure changes nothing in the images

# Every configuration that existed before: the exact structures in both radius
# modes with and without culling, and the one-photon-per-bucket grids without
# culling.  Those three are compared at their old table size: the new default
# table load (3.3) is a deliberate change of their tables, so it is pinned for
# this check — --table-load 2 reproduces the old rule of the in-trace tables,
# and --rect-buckets fixes mlrect's in both builds.  With culling, the new mask
# hash (3.2) changes which photons compete for their buckets; v3 checks those
# statistically.  The uniform grid is "grid" to the old build, "uniform" to
# both.
#
# Two of them are not reproducible from run to run in either build: the
# original stochastic grid keeps whichever photon's write lands last, and
# mlrect's keys are seeded by the photon's slot, which an atomic hands out.
# For those, before-vs-after must look like two runs of the old build.
PINNED_MLRECT = ["--rect-buckets", 1 << 23]


def v1_cases():
    cases = []
    for accel in ("uniform", "hash", "mlhash"):
        for mode in ("global", "footprint"):
            for culling in ("off", "on"):
                cases.append({"accel": accel, "args": ["--accel", accel, "--radius-mode", mode,
                                                       "--culling", culling],
                              "new": [], "kind": "exact"})
    for mode in ("global", "footprint"):
        for culling in ("off", "on"):
            cases.append({"accel": "brute", "args": ["--accel", "brute", "--radius-mode", mode,
                                                     "--culling", culling] + SMALL,
                          "new": [], "kind": "exact"})
    for mode in ("global", "footprint"):
        common = ["--radius-mode", mode]
        cases.append({"accel": "rect", "args": ["--accel", "rectified"] + common,
                      "new": ["--table-load", 2], "kind": "exact"})
        cases.append({"accel": "stochastic", "args": ["--accel", "stochastic"] + common,
                      "new": ["--table-load", 2], "kind": "nondeterministic"})
        cases.append({"accel": "mlrect", "args": ["--accel", "mlrect"] + common + PINNED_MLRECT,
                      "new": [], "kind": "nondeterministic"})
        cases.append({"accel": "mlrect-in-trace",
                      "args": ["--accel", "mlrect", "--mlrect-in-trace"] + common + PINNED_MLRECT,
                      "new": [], "kind": "nondeterministic"})
    return cases


def check_v1(args):
    pre = renderer_dir(args.baseline)
    new = renderer_dir(BUILD)
    out = os.path.join(OUT, "v1")
    frames, seed = args.frames, args.seed
    results, ok_all = [], True
    print("V1: %d frames, seed %d; before = %s" % (frames, seed, pre))
    for case in v1_cases():
        common = case["args"] + ["--frames", frames, "--seed", seed]
        tag = tag_of(common)
        a, _, _ = render(pre, out, "before_a_" + tag, common)
        b, _, _ = render(pre, out, "before_b_" + tag, common)
        n, _, _ = render(new, out, "after_" + tag, common + case["new"])
        after = compare(a, n)
        floor = compare(a, b)
        if case["kind"] == "nondeterministic":
            # the same kind of difference as the old binary has from itself
            ok = after["rel_l2"] <= 2.0 * floor["rel_l2"] + 1e-6
        else:
            ok = same_image(after)
        ok_all &= ok
        results.append({"case": " ".join(str(c) for c in case["args"] + case["new"]),
                        "kind": case["kind"], "after_vs_before": after,
                        "run_to_run": floor, "ok": ok})

    print("\n%-58s %9s %9s %9s | %9s %9s  %s" % (
        "configuration", "max abs", "max rel", "rel L2", "floor rel", "floor L2", "verdict"))
    for r in results:
        a, f = r["after_vs_before"], r["run_to_run"]
        print("%-58s %9s %9s %9s | %9s %9s  %s%s" % (
            r["case"][:58], fmt(a["max_abs"]), fmt(a["max_rel"]), fmt(a["rel_l2"]),
            fmt(f["max_rel"]), fmt(f["rel_l2"]), "PASS" if r["ok"] else "FAIL",
            "" if r["kind"] == "exact" else "  (L2 vs floor)"))
    print("(after vs before; floor = two runs of the old binary; reproducible: max rel <= 1e-5"
          " or all values allclose(1e-5, 1e-8); not reproducible: rel L2 <= 2 x floor)")
    write_results(out, "results.json", results)

    if not args.no_timing:
        ok_all &= check_timing(args)
    print("\nV1 %s" % ("PASS" if ok_all else "FAIL"))
    return ok_all


# --------------------------------------------------------------------------
# Timing: per-pass times before and after, and what moved them

TIMING_CONFIGS = (
    ("uniform/global", ["--accel", "uniform"]),
    ("hash/global", ["--accel", "hash"]),
    ("hash/footprint", ["--accel", "hash", "--radius-mode", "footprint"]),
    ("mlhash/footprint", ["--accel", "mlhash", "--radius-mode", "footprint"]),
    ("mlhash/footprint+culling", ["--accel", "mlhash", "--radius-mode", "footprint",
                                  "--culling", "on"]),
    ("rect/global", ["--accel", "rectified"]),
    ("rect/footprint", ["--accel", "rectified", "--radius-mode", "footprint"]),
    ("mlrect/footprint", ["--accel", "mlrect", "--radius-mode", "footprint"]),
    ("mlrect/footprint+culling", ["--accel", "mlrect", "--radius-mode", "footprint",
                                  "--culling", "on"]),
)

# Before -> after in three steps, so that each change gets its own column: the
# old build (OptiX validation always on, one pipeline, 3x3x3 blocks, two
# buckets per entry), the new build with validation on and the old table load,
# then validation off, then the new default table load.
TIMING_VARIANTS = (
    ("before", "pre", []),
    ("new,val on,load 2", "new", ["--validation", "on", "--table-load", 2]),
    ("new,val off,load 2", "new", ["--validation", "off", "--table-load", 2]),
    ("new (defaults)", "new", []),
)


def check_timing(args):
    bins = {"pre": renderer_dir(args.baseline), "new": renderer_dir(BUILD)}
    out = os.path.join(OUT, "timing")
    frames, warmup, runs = args.timing_frames, args.timing_warmup, args.timing_runs
    print("\nTiming: %d frames (%d warm-up), median of %d runs, 1280x720, 1M paths, depth 8, seed %d"
          % (frames, warmup, runs, args.seed))
    rows = []
    for label, extra in TIMING_CONFIGS:
        row = {"config": label, "variants": {}}
        for vname, which, vextra in TIMING_VARIANTS:
            per_run = []
            for k in range(runs):
                common = extra + BENCH + ["--frames", frames, "--warmup", warmup, "--seed", args.seed]
                name = "%s_%s_run%d" % (tag_of(extra), tag_of([vname]), k)
                _, tj, _ = render(bins[which], out, name, common + vextra)
                per_run.append(timing_means(tj, warmup))
            row["variants"][vname] = {key: median([r[key] for r in per_run]) for key in per_run[0]}
        rows.append(row)

    cols = ("camera_ms", "mark_ms", "photon_trace_ms", "build_ms", "query_ms", "accum_ms", "total_ms")
    for row in rows:
        print("\n%s" % row["config"])
        print("  %-20s" % "" + "".join("%9s" % c.replace("_ms", "").replace("photon_trace", "photon")
                                       for c in cols) + "  %9s" % "T")
        for vname, _, _ in TIMING_VARIANTS:
            m = row["variants"][vname]
            print("  %-20s" % vname + "".join("%9.3f" % m.get(c, 0.0) for c in cols)
                  + "  %9.0f" % m.get("buckets", 0))
    print("\n(ms per frame; before = the old build, whose camera pass also marks culling cells"
          " and which always runs OptiX validation; T = buckets)")

    # The kernels' registers and the environment, from the new build's record.
    with open(tj) as f:
        record = json.load(f)
    kernels = record["kernels"]
    env = record.get("environment", {})
    print("\nnew build: %s (%s, OptiX validation per variant); before: build/pre_families/build"
          " (Release)" % (BUILD, env.get("build_type")))
    if env.get("build_type") != "Release":
        print("WARNING: the new build is not a Release build; these times are not comparable")
    print("\nregisters / local bytes per thread of the CUDA kernels (new build):")
    for name, k in kernels.items():
        print("  %-24s %4d  %5d" % (name, k["registers"], k["local_bytes"]))
    print("(the OptiX programs' only show in Nsight Compute, which needs GPU performance"
          " counter access: sudo ncu --metrics launch__registers_per_thread ./renderer ...)")
    write_results(out, "results.json", {"rows": rows, "kernels": kernels})
    return True


# --------------------------------------------------------------------------
# V2 — range-based visits

GATHER_LINE = re.compile(
    r"\[Verify\] frame (\d+)\s+(OK|FAIL)\s+(\S+) vs (\S+): max abs diff = (\S+)"
    r"\s+max rel = (\S+)\s+pixels off = (\d+)")


def verify_gather(bin_dir, out, name, args):
    """One --verify-gather render: per-frame records and the overall verdict."""
    _, _, stdout = render(bin_dir, out, name, ["--verify-gather"] + args)
    frames = []
    for line in stdout.replace("\r", "\n").splitlines():
        m = GATHER_LINE.search(line)
        if m:
            frames.append({"frame": int(m.group(1)), "ok": m.group(2) == "OK",
                           "against": m.group(4), "max_abs": float(m.group(5)),
                           "max_rel": float(m.group(6)), "pixels_off": int(m.group(7))})
    return frames, "[Verify] PASS" in stdout


def ppm_scale(alpha, frame_index):
    """r_i / r_1 of the PPM schedule, computed here independently of the
    renderer: (r_i/r_1)^2 = G(i+a) G(2) / (G(1+a) G(i+1)), i = frame_index+1."""
    i = frame_index + 1
    if i <= 1:
        return 1.0
    log_ratio = (math.lgamma(i + alpha) - math.lgamma(1 + alpha)) \
        - (math.lgamma(i + 1.0) - math.lgamma(2.0))
    return math.exp(0.5 * log_ratio)


# Frames 1, 100 and 1000 (indices 0, 99, 999): the PPM factor s is 1, about
# 0.46 and about 0.32 there, so the per-pixel radii, and with them the levels
# in use, differ.  --first-frame renders exactly those frames of a long run.
FIRST_FRAMES = (0, 99, 999)

# The structures whose visits changed, with what pins them to the old table
# (see V1) and whether their image is reproducible.
V2_STRUCTURES = (
    ("uniform", ["--accel", "uniform"], [], True),
    ("hash", ["--accel", "hash"], [], True),
    ("rect", ["--accel", "rectified"], ["--table-load", 2], True),
    ("stochastic", ["--accel", "stochastic"], ["--table-load", 2], False),
    ("mlhash", ["--accel", "mlhash"], [], True),
    ("mlrect", ["--accel", "mlrect"] + PINNED_MLRECT, [], False),
)


def check_v2(args):
    pre = renderer_dir(args.baseline)
    new = renderer_dir(BUILD)
    out = os.path.join(OUT, "v2")
    seed, frames = args.seed, args.frames
    ok_all = True
    rows = []
    print("V2a: lookups before (3x3x3 cells) and after (the cells the sphere overlaps),"
          " --stats, %d frames, seed %d" % (frames, seed))
    for label, accel_args, new_extra, reproducible in V2_STRUCTURES:
        for mode in ("global", "footprint"):
            common = accel_args + ["--radius-mode", mode, "--frames", frames, "--seed", seed,
                                   "--stats", "--warmup", 0]
            tag = tag_of(common)
            b_exr, b_tj, _ = render(pre, out, "before_" + tag, common)
            a_exr, a_tj, _ = render(new, out, "after_" + tag, common + new_extra)
            before, after = timing_means(b_tj, 0), timing_means(a_tj, 0)
            d = compare(b_exr, a_exr)
            # the same pairs: every frame's mean, 95th percentile and maximum of
            # the photons accepted per lookup agree
            same_accepted = all(abs(before[k] - after[k]) <= 1e-9 * max(1.0, before[k])
                                for k in ("accepted_mean", "accepted_p95", "accepted_max"))
            ok = (same_image(d) and same_accepted) if reproducible else True
            ok_all &= ok
            rows.append({"structure": label, "mode": mode, "before": before, "after": after,
                         "image": d, "reproducible": reproducible, "ok": ok})
    print("\n%-11s %-10s %8s %9s %9s %10s %10s %9s  %s" % (
        "structure", "radius", "cells", "tested", "tested", "accepted", "accepted", "image",
        "verdict"))
    print("%-11s %-10s %8s %9s %9s %10s %10s %9s" % (
        "", "", "after", "before", "after", "before", "after", "max rel"))
    for r in rows:
        b, a = r["before"], r["after"]
        print("%-11s %-10s %8.2f %9.2f %9.2f %10.3f %10.3f %9s  %s" % (
            r["structure"], r["mode"], a["cells_mean"], b["tested_mean"], a["tested_mean"],
            b["accepted_mean"], a["accepted_mean"], fmt(r["image"]["max_rel"]),
            ("PASS" if r["ok"] else "FAIL") if r["reproducible"]
            else "(not reproducible; reported only)"))
    print("(means per lookup over all frames; before = 27 cells for every lookup; tested ="
          " photon records read, or representatives for the one-per-bucket grids)")

    print("\nV2b: exact structures against brute force at frames 1, 100, 1000 (--verify-gather)")
    vrows = []
    for accel in ("uniform", "hash", "mlhash"):
        for mode in ("global", "footprint"):
            for first in FIRST_FRAMES:
                name = "verify_%s_%s_first%d_s%d" % (accel, mode, first, seed)
                recs, passed = verify_gather(new, out, name, [
                    "--accel", accel, "--radius-mode", mode, "--first-frame", first,
                    "--frames", 2, "--seed", seed] + SMALL)
                ok = passed and len(recs) == 2
                ok_all &= ok
                vrows.append({"accel": accel, "mode": mode, "frame": first + 1, "ok": ok,
                              "worst_max_rel": max((x["max_rel"] for x in recs), default=float("nan"))})
    print("\n%-8s %-10s %7s %14s  %s" % ("struct", "radius", "frame", "worst max rel", "verdict"))
    for r in vrows:
        print("%-8s %-10s %7d %14s  %s" % (r["accel"], r["mode"], r["frame"],
                                           fmt(r["worst_max_rel"]), "PASS" if r["ok"] else "FAIL"))
    write_results(out, "results.json", {"stats": rows, "verify_gather": vrows})
    print("\nV2 %s" % ("PASS" if ok_all else "FAIL"))
    return ok_all


# --------------------------------------------------------------------------
# V3 — the culling mask is independent of the tables

VERIFY_LINE = re.compile(
    r"\[Verify\] frame (\d+)\s+(OK|FAIL)\s+(\S+) culled vs full: max abs = (\S+)"
    r"\s+max rel = (\S+)\s+\| full vs full: max rel = (\S+)\s+\| culled (\S+)% "
    r"of (\d+) deposits, levels (\d+)")


def verify_culling(bin_dir, out, name, args):
    """One --verify-culling render; per-frame records and the overall verdict."""
    _, _, stdout = render(bin_dir, out, name, ["--verify-culling"] + args)
    frames = []
    for line in stdout.replace("\r", "\n").splitlines():
        m = VERIFY_LINE.search(line)
        if m:
            frames.append({"frame": int(m.group(1)), "ok": m.group(2) == "OK",
                           "max_abs": float(m.group(4)), "max_rel": float(m.group(5)),
                           "floor_max_rel": float(m.group(6)),
                           "culled": float(m.group(7)) / 100.0,
                           "deposits": int(m.group(8)), "levels": int(m.group(9))})
    passed = "[Verify] culling PASS" in stdout
    return frames, passed


def decay_exponent(ms, values):
    """Least-squares slope of log(value) against log(M)."""
    pts = [(math.log(m), math.log(v)) for m, v in zip(ms, values) if v > 0]
    if len(pts) < 2:
        return float("nan")
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    return sxy / sxx if sxx > 0 else float("nan")


# Averages of M frames of the one-photon-per-bucket grids against their exact
# counterpart over the very same photon maps (same seed: every frame traces the
# same photons from the same gather points in all of them), with culling off
# and on.  Against that reference the photon noise cancels and what is left is
# the one-photon-per-bucket noise alone, which an unbiased grid averages away
# like 1/sqrt(M).  A culled photon is within r of no lookup, so it only ever
# added a zero to its bucket's sum: culling changes the noise, not the
# expectation.  --no-ppm with a per-pixel radius keeps every frame's radii the
# same (under the schedule each frame is noisier than the one before, which
# flattens the fall), and in footprint mode the adaptive radius stays off.
SERIES = (1, 4, 16, 64, 256)
SERIES_SETUP = ["--no-ppm", "--radius-mode", "footprint"]


def series_against_reference(bin_dir, out, seed, ref_accel, tested, common_extra, summary,
                             variants=("off", "on")):
    """Averages of M frames of each tested structure, with culling off and on
    (`variants`), against the exact `ref_accel` averaged over the same M
    frames.  Prints one table per structure and returns whether every curve
    falls like 1/sqrt(M) (L2; L1 over M >= 16, see below)."""
    ok_all = True
    refs = {}
    for m in SERIES:
        refs[m], _, _ = render(bin_dir, out, "%s_ref_f%d_s%d" % (ref_accel, m, seed),
                               ["--accel", ref_accel, "--frames", m, "--seed", seed] + common_extra)
    for accel in tested:
        rows = []
        for m in SERIES:
            common = ["--accel", accel, "--frames", m, "--seed", seed] + common_extra
            row = {"M": m}
            for v in variants:
                img, _, _ = render(bin_dir, out, "%s_%s_f%d_s%d" % (accel, v, m, seed),
                                   common + (["--culling", "on"] if v == "on" else []))
                row[v] = compare(refs[m], img)
            rows.append(row)
        summary.setdefault("series", {})[accel] = rows

        ms = [r["M"] for r in rows]
        names = {"off": "culling off", "on": "culling on"}
        print("\n%s against %s, averages of M frames (relative L1 / L2):" % (accel, ref_accel))
        print("%6s" % "M" + "".join("%24s" % names[v] for v in variants))
        for r in rows:
            print("%6d" % r["M"] + "".join("%11s / %-11s" % (fmt(r[v]["rel_l1"]), fmt(r[v]["rel_l2"]))
                                          for v in variants))
        fits = {}
        for key in variants:
            l2 = decay_exponent(ms, [r[key]["rel_l2"] for r in rows])
            # L1 of a difference made of rare spikes stays flat until most
            # pixels have collected a few of them, so its slope is taken
            # where M is large enough for that.
            big = [r for r in rows if r["M"] >= 16]
            l1 = decay_exponent([r["M"] for r in big], [r[key]["rel_l1"] for r in big])
            fits[key] = {"l1": l1, "l2": l2}
        print("fitted slopes " + "   ".join("%s: L2 M^%.2f, L1 (M>=16) M^%.2f" % (
            names[k], fits[k]["l2"], fits[k]["l1"]) for k in variants) + "   (1/sqrt(M) = -0.50)")
        summary.setdefault("series_fit", {})[accel] = fits
        ok = all(-0.65 <= fits[k]["l2"] <= -0.35 and -0.75 <= fits[k]["l1"] <= -0.3
                 for k in variants)
        print("%s: %s" % (accel, "PASS" if ok else "FAIL"))
        ok_all &= ok
    return ok_all


def check_v3(args):
    pre = renderer_dir(args.baseline)
    new = renderer_dir(BUILD)
    out = os.path.join(OUT, "v3")
    seed = args.seed
    ok_all = True
    summary = {"exact": []}

    if args.part in ("all", "a"):
        print("V3a: culled vs full map, every frame (--verify-culling), seed %d" % seed)
        for accel, extra, frames in (("brute", SMALL, 8), ("uniform", [], 16), ("hash", [], 16),
                                     ("mlhash", [], 16)):
            for mode in ("global", "footprint"):
                tag = "%s_%s_f%d_s%d" % (accel, mode, frames, seed)
                recs, passed = verify_culling(new, out, "verify_" + tag, [
                    "--accel", accel, "--radius-mode", mode, "--frames", frames,
                    "--seed", seed] + extra)
                worst = max((r["max_rel"] for r in recs), default=float("nan"))
                floor = max((r["floor_max_rel"] for r in recs), default=float("nan"))
                culled = sum(r["culled"] for r in recs) / max(len(recs), 1)
                ok = passed and len(recs) == frames and worst <= 1e-5
                ok_all &= ok
                summary["exact"].append({"accel": accel, "mode": mode, "frames": recs,
                                         "worst_max_rel": worst, "floor_max_rel": floor,
                                         "mean_culled": culled, "ok": ok})
        print("\n%-8s %-10s %7s %14s %14s %9s  %s" % ("struct", "radius", "frames", "worst max rel",
                                                     "floor max rel", "culled", "verdict"))
        for r in summary["exact"]:
            print("%-8s %-10s %7d %14s %14s %8.1f%%  %s" % (
                r["accel"], r["mode"], len(r["frames"]), fmt(r["worst_max_rel"]),
                fmt(r["floor_max_rel"]), 100 * r["mean_culled"], "PASS" if r["ok"] else "FAIL"))
        print("(floor = two full maps of the same frame, i.e. float summation order alone)")

    if args.part in ("all", "b"):
        print("\nV3b: entries tested per lookup, mlhash/footprint, culling off and on, --stats,"
              " benchmark settings, 16 frames")
        rows = []
        for which, bin_dir in (("before", pre), ("after", new)):
            for culling in ("off", "on"):
                common = ["--accel", "mlhash", "--radius-mode", "footprint", "--culling", culling,
                          "--frames", 16, "--seed", seed, "--stats", "--warmup", 1] + BENCH
                _, tj, _ = render(bin_dir, out, "tested_%s_culling-%s" % (which, culling), common)
                rows.append({"build": which, "culling": culling, "m": timing_means(tj, 1)})
        print("\n%-7s %-8s %9s %10s %9s %10s %12s %10s" % ("build", "culling", "tested", "accepted",
                                                        "culled", "stored", "entries", "T"))
        for r in rows:
            m = r["m"]
            print("%-7s %-8s %9.2f %10.3f %8.1f%% %10.0f %12.0f %10.0f" % (
                r["build"], r["culling"], m["tested_mean"], m["accepted_mean"],
                100 * m.get("culled", 0.0), m["photons"], m["entries"], m["buckets"]))
        get = {(r["build"], r["culling"]): r["m"] for r in rows}
        a_off, a_on = get[("after", "off")]["tested_mean"], get[("after", "on")]["tested_mean"]
        same_accepted = abs(get[("after", "off")]["accepted_mean"]
                            - get[("after", "on")]["accepted_mean"]) < 1e-6
        ok = a_on <= 1.05 * a_off and same_accepted
        print("after: tested with culling / without = %.3f (before: %.3f)  %s" % (
            a_on / a_off, get[("before", "on")]["tested_mean"] / get[("before", "off")]["tested_mean"],
            "PASS" if ok else "FAIL"))
        ok_all &= ok
        summary["tested"] = rows

    if args.part in ("all", "c"):
        print("\nV3c: one-photon-per-bucket grids against the exact ones, culling off (A) and"
              " on (B), --no-ppm, footprint radius, seed %d" % seed)
        ok_all &= series_against_reference(new, out, seed, "mlhash", ["mlrect"], SERIES_SETUP, summary)
        ok_all &= series_against_reference(new, out, seed, "hash", ["rect"], SERIES_SETUP, summary)

    write_results(out, "results_%s.json" % args.part, summary)
    print("\nV3 %s" % ("PASS" if ok_all else "FAIL"))
    return ok_all


# --------------------------------------------------------------------------
# V4 — splatting finds exactly what the exact gathers find

def check_v4(args):
    """Every frame of splat against hash and of mlsplat against mlhash, at
    frames 1, 100 and 1000 (the PPM factor 1, 0.46, 0.32: other radii, other
    levels), both radius modes, cell factor F = 1 and 2: --verify-gather runs
    the splatting frame and then the exact gather over the same gather points
    and photon paths.  The same pairs adding the same products, the images
    differ by the order of the sums alone: max rel <= 1e-5.  Four
    accumulator copies (--splat-stripes 4) once more on the hardest case."""
    new = renderer_dir(BUILD)
    out = os.path.join(OUT, "v4")
    seed = args.seed
    cases = []
    for accel in ("splat", "mlsplat"):
        for mode in ("global", "footprint"):
            for factor in (1, 2):
                for first in FIRST_FRAMES:
                    cases.append((accel, mode, factor, 1, first))
    for first in FIRST_FRAMES:
        cases.append(("mlsplat", "footprint", 2, 4, first))
    rows, ok_all = [], True
    print("V4: splatting against the exact gather, per frame (--verify-gather), benchmark settings,"
          " %d frames from each start, seed %d" % (args.frames, seed))
    for accel, mode, factor, stripes, first in cases:
        name = "verify_%s_%s_F%d_K%d_first%d_s%d" % (accel, mode, factor, stripes, first, seed)
        recs, passed = verify_gather(new, out, name, [
            "--accel", accel, "--radius-mode", mode, "--splat-cell-factor", factor,
            "--splat-stripes", stripes, "--first-frame", first, "--frames", args.frames,
            "--seed", seed] + BENCH)
        worst = max((r["max_rel"] for r in recs), default=float("nan"))
        ok = passed and len(recs) == args.frames and worst <= 1e-5
        ok_all &= ok
        rows.append({"accel": accel, "mode": mode, "F": factor, "stripes": stripes,
                     "frame": first + 1, "against": recs[0]["against"] if recs else "?",
                     "worst_max_rel": worst, "worst_max_abs": max((r["max_abs"] for r in recs),
                                                                  default=float("nan")),
                     "ok": ok})
    print("\n%-8s %-10s %3s %3s %6s %-7s %14s %14s  %s" % (
        "accel", "radius", "F", "K", "frame", "against", "worst max rel", "worst max abs", "verdict"))
    for r in rows:
        print("%-8s %-10s %3d %3d %6d %-7s %14s %14s  %s" % (
            r["accel"], r["mode"], r["F"], r["stripes"], r["frame"], r["against"],
            fmt(r["worst_max_rel"]), fmt(r["worst_max_abs"]), "PASS" if r["ok"] else "FAIL"))
    write_results(out, "results.json", rows)
    print("\nV4 %s" % ("PASS" if ok_all else "FAIL"))
    return ok_all


# --------------------------------------------------------------------------
# V5 — the stochastic structures are unbiased

def check_v5(args):
    """Averages of M = 1..256 frames of each stochastic structure against its
    exact counterpart over the same photon maps (same seed, --no-ppm, a
    per-pixel radius): rect against hash and mlrect against mlhash, as built
    now (the default table load), and stochastic splatting against mlsplat.
    An unbiased one closes in like 1/sqrt(M)."""
    new = renderer_dir(BUILD)
    out = os.path.join(OUT, "v5")
    seed = args.seed
    summary, ok_all = {}, True
    print("V5: averages of M frames against the exact counterpart, --no-ppm, footprint radius,"
          " seed %d" % seed)
    ok_all &= series_against_reference(new, out, seed, "hash", ["rect"], SERIES_SETUP, summary,
                                       variants=("off",))
    ok_all &= series_against_reference(new, out, seed, "mlhash", ["mlrect"], SERIES_SETUP, summary,
                                       variants=("off",))
    ok_all &= series_against_reference(new, out, seed, "mlsplat", ["stochsplat"], SERIES_SETUP,
                                       summary, variants=("off",))
    write_results(out, "results.json", summary)
    print("\nV5 %s" % ("PASS" if ok_all else "FAIL"))
    return ok_all


# --------------------------------------------------------------------------
# V6 — fused family A renders what the split one renders

def check_v6(args):
    """--fused against the split camera and query passes, exact structures,
    both radius modes: the fused frames build their structure from the frame
    before's radii and levels, which changes how photons are found, never
    which — the images must be the same up to float noise."""
    new = renderer_dir(BUILD)
    out = os.path.join(OUT, "v6")
    seed, frames = args.seed, args.frames
    rows, ok_all = [], True
    print("V6: fused against split, %d frames, seed %d, benchmark settings" % (frames, seed))
    for accel in ("uniform", "hash", "mlhash"):
        for mode in ("global", "footprint"):
            common = ["--accel", accel, "--radius-mode", mode, "--frames", frames,
                      "--seed", seed] + BENCH
            tag = tag_of(common)
            split, tj_s, _ = render(new, out, "split_" + tag, common)
            fused, tj_f, _ = render(new, out, "fused_" + tag, common + ["--fused"])
            floor_img, _, _ = render(new, out, "split2_" + tag, common)
            d = compare(split, fused)
            floor = compare(split, floor_img)
            ok = same_image(d)
            ok_all &= ok
            rows.append({"accel": accel, "mode": mode, "image": d, "floor": floor, "ok": ok,
                         "split": timing_means(tj_s, 1), "fused": timing_means(tj_f, 1)})
    print("\n%-8s %-10s %10s %10s %10s  %s" % ("struct", "radius", "max rel", "floor rel",
                                                "rel L2", "verdict"))
    for r in rows:
        print("%-8s %-10s %10s %10s %10s  %s" % (r["accel"], r["mode"], fmt(r["image"]["max_rel"]),
                                                fmt(r["floor"]["max_rel"]), fmt(r["image"]["rel_l2"]),
                                                "PASS" if r["ok"] else "FAIL"))
    print("(fused vs split; floor = two split runs)")
    write_results(out, "results.json", rows)
    print("\nV6 %s" % ("PASS" if ok_all else "FAIL"))
    return ok_all


# --------------------------------------------------------------------------
# V7 — error against time falls at the Knaus-Zwicker rate

def check_v7(args):
    """Exact configurations on the Cornell box against the path-traced
    reference: the slope of log relMSE against log GPU time, over the
    checkpoints from 16 frames on, must be close to -2/3 (within 0.53..0.8),
    the rate of probabilistic PPM at alpha = 2/3.

    Judged: the per-pixel radius (A and B, k_pix 4), and the global radius
    started at r1 = 0.5% of the scene radius.  The global radius at its default
    r1 = 1.5% is reported, not judged: in 1024 frames its error is still
    dominated by the bias of the large early radii, which shrinks more slowly
    than r^2 near the box's walls and edges, so its curve flattens towards the
    end (the rate is reached with the smaller r1, which is the point)."""
    out = os.path.join(OUT, "v7")
    tools = os.path.join(ROOT, "eval")
    print("V7: reference (eval/render_references.py), then the runs (eval/error_vs_time.py)")
    r = subprocess.run([sys.executable, os.path.join(tools, "render_references.py"), "--scenes",
                        "cornell", "--frames", str(args.ref_frames), "--parts", "2"])
    if r.returncode != 0:
        fail("render_references.py failed")
    groups = (("default", ["A/hash/global", "A/mlhash/footprint", "B/mlhash/footprint"], ""),
              ("r1 0.5%", ["A/hash/global"], "--radius 0.005"))
    series = []
    for label, configs, extra in groups:
        sub_out = os.path.join(out, label.replace(" ", "_").replace("%", "pct"))
        r = subprocess.run([sys.executable, os.path.join(tools, "error_vs_time.py"), "--scene",
                            "cornell", "--render", "--out", sub_out, "--frames", str(args.frames),
                            "--seeds", str(args.seed), "--k-pix", "4", "--extra", extra,
                            "--configs"] + configs)
        if r.returncode != 0:
            fail("error_vs_time.py failed")
        with open(os.path.join(sub_out, "cornell", "error_vs_time.json")) as f:
            summary = json.load(f)
        for s in summary["series"]:
            series.append((label, s))
    ok_all = True
    print("\n%-22s %-9s %8s %10s  %s" % ("configuration", "radius", "slope", "relMSE", "verdict"))
    for label, s in series:
        judged = not (label == "default" and s["config"] == "A/hash/global")
        ok = -0.8 <= s["slope"] <= -0.53
        if judged:
            ok_all &= ok
        radius = ("k_pix 4" if s["config"].endswith("footprint")
                  else "r1 1.5%" if label == "default" else "r1 0.5%")
        print("%-22s %-9s %8.3f %10.3g  %s" % (
            s["config"], radius, s["slope"], s["points"][-1][2],
            ("PASS" if ok else "FAIL") if judged else "(reported: bias-dominated at the default r1)"))
    print("(slope of log relMSE against log GPU time from 16 frames on; -2/3 expected; the"
          " reference's own relMSE is %.2g)" % summary["reference_relmse"])
    print("\nV7 %s" % ("PASS" if ok_all else "FAIL"))
    return ok_all


def check_diff(args):
    d = compare(args.a, args.b)
    for k, v in d.items():
        print("%-18s %s" % (k, v))
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="check", required=True)

    def timing_options(q):
        q.add_argument("--timing-frames", type=int, default=128)
        q.add_argument("--timing-warmup", type=int, default=8)
        q.add_argument("--timing-runs", type=int, default=3)

    v1 = sub.add_parser("v1", help="infrastructure: images unchanged, timings, registers")
    v1.add_argument("--baseline", default=PRE, help="directory with the old renderer and device.ptx")
    v1.add_argument("--frames", type=int, default=32)
    v1.add_argument("--seed", type=int, default=1)
    v1.add_argument("--no-timing", action="store_true", help="images only")
    timing_options(v1)

    t = sub.add_parser("timing", help="per-pass times before and after")
    t.add_argument("--baseline", default=PRE)
    t.add_argument("--seed", type=int, default=1)
    timing_options(t)

    v2 = sub.add_parser("v2", help="range-based visits: images, cells and candidates per lookup")
    v2.add_argument("--baseline", default=PRE)
    v2.add_argument("--frames", type=int, default=8)
    v2.add_argument("--seed", type=int, default=1)

    v3 = sub.add_parser("v3", help="mask decoupling: culling exact, tested entries, unbiased")
    v3.add_argument("--baseline", default=PRE)
    v3.add_argument("--seed", type=int, default=1)
    v3.add_argument("--part", choices=("all", "a", "b", "c"), default="all",
                    help="a = culled vs full per frame, b = tested entries, c = averages")

    v4 = sub.add_parser("v4", help="splat/mlsplat equal hash/mlhash per frame")
    v4.add_argument("--seed", type=int, default=1)
    v4.add_argument("--frames", type=int, default=2)

    v5 = sub.add_parser("v5", help="rect, mlrect, stochsplat: unbiased (1/sqrt(M))")
    v5.add_argument("--seed", type=int, default=1)

    v6 = sub.add_parser("v6", help="fused family A: the same images as split")
    v6.add_argument("--seed", type=int, default=1)
    v6.add_argument("--frames", type=int, default=32)

    v7 = sub.add_parser("v7", help="error against time: slope -2/3 on the Cornell box")
    v7.add_argument("--seed", type=int, default=1)
    v7.add_argument("--frames", type=int, default=1024)
    v7.add_argument("--ref-frames", type=int, default=2048, help="frames per part of the reference")

    d = sub.add_parser("diff", help="compare two EXRs")
    d.add_argument("a")
    d.add_argument("b")

    args = p.parse_args()
    ok = {"v1": check_v1, "timing": check_timing, "v2": check_v2, "v3": check_v3,
          "v4": check_v4, "v5": check_v5, "v6": check_v6, "v7": check_v7,
          "diff": check_diff}[args.check](args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
