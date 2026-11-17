"""Shared by the evaluation scripts: the scenes, running the renderer, EXR
files and the error measure.  Pure Python — nothing here needs numpy.

The EXR reader is the one checks.py uses (uncompressed scanline files, which
is what the renderer writes); the writer below produces the same kind.
"""

import json
import os
import struct
import subprocess
import sys
from array import array

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from checks import BUILD, read_exr  # noqa: E402  (BUILD: build-release/ unless RENDERER_BUILD)

RENDERER = os.path.join(BUILD, "renderer")

# The renderer needs the discrete GPU even in offline mode.
RENDER_ENV = dict(os.environ)
RENDER_ENV["__NV_PRIME_RENDER_OFFLOAD"] = "1"
RENDER_ENV["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

# Scene name -> (glTF file, camera file), both resolved by the renderer in
# scenes/.  The Cornell box is the original scene.glb.
SCENES = {
    "cornell": ("scene.glb", "camera.txt"),
    "stadium": ("stadium.glb", "stadium_camera.txt"),
    "rooms": ("rooms.glb", "rooms_camera.txt"),
    "caustic": ("caustic.glb", "caustic_camera.txt"),
}


# A configuration is <family>/<accel>/<radius mode>[+variant...], as the
# renderer names it in its logs:
#
#   A  gather from the full photon map            A/hash/global
#   B  gather from a culled map (--culling on)    B/mlhash/footprint
#   C  splatting                                  C/mlsplat/footprint
#   D  adaptive LoD hash-grid SPPM                D/sppm_lepski/footprint
#   E  RTPM, photons in acceleration structures   E/rtpm/global
#
# and a variant sets one option of the renderer.
FAMILIES = ("A", "B", "C", "D", "E")

# Variants without a value, and those that are a prefix followed by one
# (longest prefix first, so that "kpix8" is not read as "k" + "pix8").
VARIANT_FLAGS = {
    "inline": ["--inline-positions"],
    "hier": ["--hier-mask"],
    "fused": ["--fused"],
    "cull": ["--rtpm-culling", "on"],          # RTPM's Photon Culling
    "uniform": ["--rtpm-kernel", "uniform"],   # RTPM with the paper's kernel
}
VARIANT_VALUES = (
    ("kappa", "--lepski-kappa"),       # kappa2
    ("kpix", "--k-pix"),               # kpix8: radius in pixel footprints
    ("reject", "--rtpm-reject"),       # reject0.7: rejection of global photons
    ("stoch", "--rtpm-stochastic"),    # stoch3: Stochastic Evaluation's list
    ("caustic", "--rtpm-caustic-scale"),
    ("r1_", "--radius"),               # r1_0.005: initial radius, of the scene radius
    ("paths", "--paths"),              # paths300000: photon paths per frame
    ("h", "--lepski-h0"),              # h0.08: level-0 cell, of d0
    ("F", "--splat-cell-factor"),      # F1
    ("K", "--splat-stripes"),          # K4
)


def split_config(label):
    """(family, accel, mode, [variants]) of a configuration label."""
    base, _, variant = label.partition("+")
    try:
        family, accel, mode = base.split("/")
    except ValueError:
        sys.exit("configuration %r is not <family>/<accel>/<radius mode>[+variant...]" % label)
    if family not in FAMILIES or mode not in ("global", "footprint"):
        sys.exit("configuration %r: family %s, radius mode global or footprint"
                 % (label, ", ".join(FAMILIES)))
    return family, accel, mode, [v for v in variant.split("+") if v]


def config_options(label):
    """Renderer options of a configuration label (without --photon)."""
    family, accel, mode, variants = split_config(label)
    opts = ["--accel", accel, "--radius-mode", mode]
    # C culls implicitly, D with a bitset of its own, E with the paper's mask
    # (+cull): only the gathering families take the renderer's --culling.
    if family in ("A", "B"):
        opts += ["--culling", "on" if family == "B" else "off"]
    for v in variants:
        if v in VARIANT_FLAGS:
            opts += VARIANT_FLAGS[v]
            continue
        for prefix, flag in VARIANT_VALUES:
            value = v[len(prefix):]
            if v.startswith(prefix) and value[:1].isdigit():
                opts += [flag, value]
                break
        else:
            sys.exit("configuration %r: unknown variant %r" % (label, v))
    return opts


def has_variant(label, prefix):
    """True if the label carries a variant that starts with `prefix`."""
    return any(v.startswith(prefix) for v in split_config(label)[3])


def run_renderer(args, log_path):
    """Runs build/renderer with `args` (cwd build/, as it resolves ../scenes);
    its output goes to log_path.  Exits on failure."""
    cmd = [RENDERER] + [str(a) for a in args]
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        r = subprocess.run(cmd, cwd=BUILD, env=RENDER_ENV, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit("render failed (exit %d), log in %s" % (r.returncode, log_path))
    return cmd


def write_exr(path, w, h, planes):
    """An uncompressed scanline EXR of 32-bit float channels B, G, R, rows in
    the order given (the renderer's own files store the top row first)."""
    chans = sorted(planes)  # EXR wants channels in alphabetical order

    def attr(name, typ, data):
        return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(data)) + data

    chlist = b"".join(c.encode() + b"\0" + struct.pack("<iB3xii", 2, 0, 1, 1) for c in chans) + b"\0"
    header = (struct.pack("<ii", 20000630, 2)
              + attr("channels", "chlist", chlist)
              + attr("compression", "compression", b"\0")
              + attr("dataWindow", "box2i", struct.pack("<iiii", 0, 0, w - 1, h - 1))
              + attr("displayWindow", "box2i", struct.pack("<iiii", 0, 0, w - 1, h - 1))
              + attr("lineOrder", "lineOrder", b"\0")
              + attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
              + attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
              + attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
              + b"\0")
    line_bytes = 8 + 4 * w * len(chans)
    first = len(header) + 8 * h
    offsets = struct.pack("<%dQ" % h, *[first + y * line_bytes for y in range(h)])
    with open(path, "wb") as f:
        f.write(header)
        f.write(offsets)
        for y in range(h):
            f.write(struct.pack("<ii", y, 4 * w * len(chans)))
            for c in chans:
                f.write(planes[c][y * w:(y + 1) * w].tobytes())


def mean_of(paths):
    """The per-pixel mean of several images of the same size."""
    acc, w, h = None, 0, 0
    for p in paths:
        w, h, planes = read_exr(p)
        if acc is None:
            acc = {c: array("d", planes[c]) for c in "RGB"}
        else:
            for c in "RGB":
                a, b = acc[c], planes[c]
                for i in range(len(a)):
                    a[i] += b[i]
    n = float(len(paths))
    return w, h, {c: array("f", (v / n for v in acc[c])) for c in "RGB"}


# The error measure: relMSE = mean over pixels and channels of
# (x - ref)^2 / (ref^2 + 0.01).  The 0.01 keeps near-black pixels, where any
# relative error is meaningless, from dominating the mean.
REL_MSE_EPS = 0.01


def ref_weights(ref_planes):
    """1 / (ref^2 + 0.01) per value, computed once per reference."""
    return {c: array("d", (1.0 / (v * v + REL_MSE_EPS) for v in ref_planes[c])) for c in "RGB"}


def rel_mse(planes, ref_planes, weights):
    total, n = 0.0, 0
    for c in "RGB":
        x, r, wt = planes[c], ref_planes[c], weights[c]
        total += sum((a - b) * (a - b) * k for a, b, k in zip(x, r, wt))
        n += len(r)
    return total / n


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
