#!/usr/bin/env python3
"""Render one converged path-traced reference plus a small grid of photon-mapped
images in a single call.

    python3 exec.py                    # reference + 4 photon renders into ./renders
    python3 exec.py --dry-run          # print the render commands only
    python3 exec.py --budget 600       # give the reference 10 minutes instead of 4
    python3 exec.py --pt-frames 3060   # fix the reference frame count, skip calibration

The reference is a path trace at --pt-depth with as many frames as fit in
--budget seconds; the frame count is picked from two short probe renders rather
than guessed, so the same budget lands on the same wall-clock time on any GPU.
The photon images are the four corners of {small,big} frames x {small,big}
depth, which is the pair of axes worth showing: frames average the density
estimate down, depth decides how much of the transport is in it at all.

Everything runs with cwd = build/, because the renderer resolves the scene and
the camera relative to ../scenes/.  Outputs are EXR, written to --out.
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "build")
RENDERER = os.path.join(BUILD, "renderer")
SRC = os.path.join(ROOT, "src")

# The renderer needs the discrete GPU even in offline mode.
RENDER_ENV = dict(os.environ)
RENDER_ENV["__NV_PRIME_RENDER_OFFLOAD"] = "1"
RENDER_ENV["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

# Two probe lengths far enough apart that the fit separates startup from the
# per-frame cost, and short enough that they cost well under a minute.
PROBE_FRAMES = (8, 32)

# The measured per-frame cost of a probe is optimistic: a long render heats the
# GPU up and clocks it down.  Budget for the slower steady state.
SLOWDOWN = 1.25


def parse_args():
    p = argparse.ArgumentParser(
        description="Render a path-traced reference and a grid of photon-mapped images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scene", default="scene.glb",
                   help="scene file, resolved by the renderer inside scenes/")
    p.add_argument("--camera", default="camera.txt",
                   help="camera file, resolved by the renderer inside scenes/")
    p.add_argument("--out", default=os.path.join(ROOT, "renders"),
                   help="directory to write the EXRs and manifest.json into")
    p.add_argument("--seed", type=int, default=1,
                   help="random seed shared by every render in the run")

    p.add_argument("--budget", type=float, default=240.0,
                   help="seconds of rendering to spend on the path-traced reference")
    p.add_argument("--pt-frames", type=int, default=0,
                   help="fixed reference frame count; skips the timing probes")
    p.add_argument("--pt-depth", type=int, default=32,
                   help="max path length for the reference")

    p.add_argument("--frames", type=int, nargs=2, default=[32, 128],
                   metavar=("SMALL", "BIG"), help="photon-mapping frame counts")
    p.add_argument("--depths", type=int, nargs=2, default=[8, 32],
                   metavar=("SMALL", "BIG"), help="photon-mapping max path lengths")
    p.add_argument("--paths", type=int, default=1_000_000,
                   help="photon paths per frame")

    p.add_argument("--sky", type=float, default=None,
                   help="sky intensity; omitted keeps the renderer default (off)")
    p.add_argument("--sky-photons", type=int, choices=(0, 1), default=None,
                   help="emit sky photons; only meaningful together with --sky")

    p.add_argument("--build", action="store_true",
                   help="run cmake --build on build/ before rendering")
    p.add_argument("--force", action="store_true",
                   help="re-render images that already exist in --out")
    p.add_argument("--skip-reference", action="store_true",
                   help="render only the photon-mapped images")
    p.add_argument("--dry-run", action="store_true",
                   help="print the render commands and the plan, render nothing")
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


def build():
    print("building ...")
    r = subprocess.run(["cmake", "--build", ".", "-j", str(os.cpu_count() or 4)],
                       cwd=BUILD)
    if r.returncode != 0:
        fail("build failed")


def render_cmd(args, *, photon, frames, depth, output, paths=None):
    cmd = [RENDERER, args.scene, "--offline",
           "--camera", args.camera,
           "--output", output,
           "--frames", str(frames),
           "--depth", str(depth),
           "--seed", str(args.seed)]
    if photon:
        cmd.append("--photon")
        cmd += ["--paths", str(paths if paths is not None else args.paths)]
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
    print("\n$ " + shell(cmd) + "   (cwd: build/)")
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


def calibrate(args, tmp_output):
    """Fit seconds = startup + per_frame * frames from two short path traces."""
    print("calibrating the reference frame count against a %s budget ..."
          % hms(args.budget))
    probes = []
    for frames in PROBE_FRAMES:
        cmd = render_cmd(args, photon=False, frames=frames, depth=args.pt_depth,
                         output=tmp_output)
        probes.append((frames, run_render(cmd, tmp_output, args.dry_run)))

    if args.dry_run:
        # Nothing ran, so there is nothing to fit; the frame count the real run
        # would pick stands in as a placeholder in the commands below.
        return "N", {"dry_run": True}

    (f_lo, t_lo), (f_hi, t_hi) = probes
    measured_per_frame = (t_hi - t_lo) / float(f_hi - f_lo)
    if measured_per_frame <= 0:
        fail("timing probes were not monotonic (%.2fs for %d frames, %.2fs for %d) — "
             "the GPU is probably busy; re-run, or pass --pt-frames" %
             (t_lo, f_lo, t_hi, f_hi))
    per_frame = measured_per_frame * SLOWDOWN
    startup = max(0.0, t_lo - f_lo * measured_per_frame)

    frames = int((args.budget - startup) / per_frame)
    frames = max(1, frames)
    info = {
        "startup": startup,
        "measured_per_frame": measured_per_frame,
        "slowdown": SLOWDOWN,
        "per_frame": per_frame,
        "probes": probes,
        "predicted": startup + frames * per_frame,
    }
    print("  startup %.2fs, %.4fs/frame (x%.2f) -> %d frames, ~%s"
          % (startup, measured_per_frame, SLOWDOWN, frames, hms(info["predicted"])))
    return frames, info


def main():
    args = parse_args()
    if args.build:
        build()
    check_environment(args)

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    tmp_output = os.path.join(args.out, ".probe.exr")

    frames_small, frames_big = sorted(args.frames)
    depth_small, depth_big = sorted(args.depths)

    manifest = {
        "scene": args.scene,
        "camera": args.camera,
        "seed": args.seed,
        "paths": args.paths,
        "budget": args.budget,
        "sky": args.sky,
        "sky_photons": args.sky_photons,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference": None,
        "photon": [],
    }

    def save_manifest():
        if args.dry_run:
            return
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    run_start = time.time()

    # ---- the reference path trace -------------------------------------------
    if not args.skip_reference:
        if args.pt_frames > 0:
            pt_frames = args.pt_frames
        else:
            pt_frames, manifest["calibration"] = calibrate(args, tmp_output)

        name = "pt_ref_f%s_d%d_s%d.exr" % (pt_frames, args.pt_depth, args.seed)
        output = os.path.join(args.out, name)
        cmd = render_cmd(args, photon=False, frames=pt_frames, depth=args.pt_depth,
                         output=output)
        entry = {"file": name, "mode": "path tracing", "frames": pt_frames,
                 "depth": args.pt_depth, "seed": args.seed, "command": shell(cmd)}
        if os.path.exists(output) and not args.force:
            print("\nreference already rendered: %s (--force to redo)" % name)
            entry["skipped"] = True
        else:
            print("\n=== reference path trace: %s frames, depth %d ==="
                  % (pt_frames, args.pt_depth))
            entry["seconds"] = run_render(cmd, output, args.dry_run)
        manifest["reference"] = entry
        save_manifest()

    # ---- the photon-mapped grid ---------------------------------------------
    # Cheapest first, so a run that gets interrupted still leaves the corners
    # that are quickest to reproduce.
    grid = [(f, d) for d in (depth_small, depth_big)
            for f in (frames_small, frames_big)]
    grid.sort(key=lambda fd: fd[0] * fd[1])

    print("\n=== %d photon-mapped renders: frames %s x depth %s, %d paths/frame ==="
          % (len(grid), (frames_small, frames_big), (depth_small, depth_big), args.paths))

    unit_costs = []
    for i, (frames, depth) in enumerate(grid):
        name = "pm_f%d_d%d_p%d_s%d.exr" % (frames, depth, args.paths, args.seed)
        output = os.path.join(args.out, name)
        cmd = render_cmd(args, photon=True, frames=frames, depth=depth, output=output)
        entry = {"file": name, "mode": "photon mapping", "frames": frames,
                 "depth": depth, "paths": args.paths, "seed": args.seed,
                 "command": shell(cmd)}
        if os.path.exists(output) and not args.force:
            print("\nalready rendered: %s (--force to redo)" % name)
            entry["skipped"] = True
        else:
            remaining = grid[i + 1:]
            if unit_costs and remaining:
                unit = sum(unit_costs) / len(unit_costs)
                eta = sum(unit * f * d for f, d in remaining)
                print("\n[%d/%d] frames %d, depth %d   (~%s left after this one)"
                      % (i + 1, len(grid), frames, depth, hms(eta)))
            else:
                print("\n[%d/%d] frames %d, depth %d" % (i + 1, len(grid), frames, depth))
            seconds = run_render(cmd, output, args.dry_run)
            entry["seconds"] = seconds
            if seconds > 0:
                unit_costs.append(seconds / float(frames * depth))
        manifest["photon"].append(entry)
        save_manifest()

    manifest["total_seconds"] = time.time() - run_start
    save_manifest()

    if os.path.exists(tmp_output):
        os.remove(tmp_output)

    if args.dry_run:
        print("\ndry run: nothing was rendered")
        return

    print("\ndone in %s — images in %s"
          % (hms(manifest["total_seconds"]), short_path(args.out)))
    for entry in ([manifest["reference"]] if manifest["reference"] else []) + manifest["photon"]:
        print("  %-40s %s" % (entry["file"],
                              "reused" if entry.get("skipped") else hms(entry["seconds"])))
    print("  %-40s render settings and timings" % "manifest.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
