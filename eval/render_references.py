#!/usr/bin/env python3
"""Path-traced reference images for the error-against-time comparisons.

    python3 eval/render_references.py                      # all four scenes
    python3 eval/render_references.py --scenes cornell --frames 4096 --parts 2

For every scene, `--parts` independent path traces of `--frames` frames each
(4 samples per pixel per frame; seeds 1001, 1002, ...), at the maximum path
depth of the photon-mapping runs, written as 32-bit float EXRs:

    eval/refs/<scene>_pt_d<D>_r<W>x<H>_f<F>_s<seed>.exr

then their mean, the reference itself, and a record of how it was made:

    eval/refs/<scene>_pt_d<D>_r<W>x<H>_f<parts*F>_ref.exr
    eval/refs/<scene>_pt_d<D>_r<W>x<H>_f<parts*F>_ref.json

The record also holds the reference's own noise: two independent parts
differ by twice the error one part has against the converged image, so
relMSE(part 1, part 2) / 2 is one part's relMSE, and the mean of P parts has
1/P of it.  An error curve that comes down to that level is measuring the
reference, not the method; error_vs_time.py draws it as a floor.

The depth must be the photon-mapping runs' (--depth): the shared path-length
budget is what makes the two solve the same problem.  Parts already rendered
with the same settings (the name says all of them) are reused; --force
renders them again.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalio  # noqa: E402

OUT = os.path.join(evalio.ROOT, "eval", "refs")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", default="cornell,stadium,rooms,caustic")
    p.add_argument("--frames", type=int, default=2048, help="frames per part (4 samples each)")
    p.add_argument("--parts", type=int, default=2, help="independent parts averaged into the reference")
    p.add_argument("--depth", type=int, default=8, help="maximum path length, as in the photon-mapping runs")
    p.add_argument("--res", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    p.add_argument("--force", action="store_true", help="render parts that already exist again")
    args = p.parse_args()
    if args.parts < 2:
        sys.exit("--parts must be at least 2: the noise estimate needs two independent parts")

    os.makedirs(OUT, exist_ok=True)
    w, h = args.res
    for scene in args.scenes.split(","):
        if scene not in evalio.SCENES:
            sys.exit("unknown scene %s (known: %s)" % (scene, ", ".join(evalio.SCENES)))
        glb, cam = evalio.SCENES[scene]
        stem = "%s_pt_d%d_r%dx%d" % (scene, args.depth, w, h)
        parts, commands, seconds = [], [], []
        for k in range(args.parts):
            seed = 1001 + k
            out = os.path.join(OUT, "%s_f%d_s%d.exr" % (stem, args.frames, seed))
            parts.append(out)
            if os.path.exists(out) and not args.force:
                print("%s part %d: %s exists" % (scene, k + 1, os.path.basename(out)))
                commands.append(None)
                seconds.append(None)
                continue
            print("%s part %d: path tracing %d frames ..." % (scene, k + 1, args.frames), flush=True)
            t0 = time.time()
            cmd = evalio.run_renderer([glb, "--offline", "--camera", cam, "--res", w, h,
                                       "--frames", args.frames, "--depth", args.depth,
                                       "--seed", seed, "--float-exr", "--output", out],
                                      out[:-4] + ".log")
            commands.append(" ".join(cmd))
            seconds.append(time.time() - t0)

        # Noise of one part, from the first two; then the mean of all parts.
        _, _, a = evalio.read_exr(parts[0])
        _, _, b = evalio.read_exr(parts[1])
        noise_part = 0.5 * evalio.rel_mse(a, b, evalio.ref_weights(b))
        rw, rh, ref = evalio.mean_of(parts)
        if (rw, rh) != (w, h):
            sys.exit("%s: parts are %dx%d, expected %dx%d" % (scene, rw, rh, w, h))
        name = "%s_f%d_ref" % (stem, args.frames * args.parts)
        evalio.write_exr(os.path.join(OUT, name + ".exr"), w, h, ref)
        evalio.save_json(os.path.join(OUT, name + ".json"), {
            "scene": scene, "glb": glb, "camera": cam, "depth": args.depth, "res": [w, h],
            "frames": args.frames * args.parts, "samples_per_pixel": 4 * args.frames * args.parts,
            "parts": [os.path.basename(x) for x in parts], "commands": commands,
            "seconds": seconds,
            "relmse_one_part": noise_part,
            "relmse_reference": noise_part / args.parts,
        })
        print("%s: reference %s.exr, %d samples per pixel; its own relMSE ~ %.3g"
              " (one part: %.3g)" % (scene, name, 4 * args.frames * args.parts,
                                     noise_part / args.parts, noise_part))


if __name__ == "__main__":
    main()
