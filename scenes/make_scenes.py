#!/usr/bin/env python3
"""Writes the test scenes of the method comparison, each with its camera file:

    python3 scenes/make_scenes.py          # all three, into scenes/

    stadium.glb  stadium_camera.txt   an indoor arena: a 240 x 240 field with
                                      tiered stands 120 units away and a roof,
                                      a detailed object 2.5 units in front of
                                      the camera, a mirror sphere and four
                                      floodlights: visible depths from 2 to
                                      over 200 (about 100x), which is what a
                                      per-pixel radius is for
    rooms.glb    rooms_camera.txt     nine rooms joined by doorways, three of
                                      them lit; the camera sees one room and a
                                      doorway, so most photons land where no
                                      lookup can reach them (culling)
    caustic.glb  caustic_camera.txt   a glass sphere on a diffuse floor next to
                                      a small bright light, and the caustic it
                                      focuses onto the floor

Follows the loader's conventions (loadScene in src/main.cpp): vertices are
written in world space, because node transforms are not applied; a material
whose name contains "mirror" or "glass" is a perfect mirror or a dielectric
of index 1.5, every other one is Lambertian with its base colour as albedo;
emissiveFactor is the emitted radiance, so it may exceed 1, and a surface
only emits on the side its normal points to, which is also the side photons
leave from.  The glass sphere's normals point outwards: the renderer tells
entering from leaving by them.  A camera file is "x y z yaw pitch", angles in
degrees, forward = (cos p cos y, sin p, cos p sin y), y up, 45 degrees of
vertical field of view.

Pure Python, no dependencies; the output is deterministic.
"""

import json
import math
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# vectors

def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a):
    l = math.sqrt(dot(a, a))
    return (a[0] / l, a[1] / l, a[2] / l) if l > 0 else (0.0, 0.0, 1.0)


# --------------------------------------------------------------------------
# geometry: one mesh per material, triangles with per-vertex normals

class Scene:
    def __init__(self):
        self.materials = []  # (name, albedo, emission)
        self.meshes = {}  # material index -> (positions, normals, indices)

    def material(self, name, albedo=(0.7, 0.7, 0.7), emission=(0.0, 0.0, 0.0)):
        for i, m in enumerate(self.materials):
            if m[0] == name:
                return i
        self.materials.append((name, albedo, emission))
        return len(self.materials) - 1

    def _mesh(self, mat):
        return self.meshes.setdefault(mat, ([], [], []))

    def tri(self, mat, p, n):
        """One triangle, vertices p[0..2] with normals n[0..2]."""
        pos, nrm, idx = self._mesh(mat)
        base = len(pos)
        pos.extend(p)
        nrm.extend(n)
        idx.extend((base, base + 1, base + 2))

    def quad(self, mat, c, u, v):
        """The parallelogram c +- u +- v; its normal is u x v, and that is the
        side it emits from if its material emits."""
        n = norm(cross(u, v))
        p = [sub(sub(c, u), v), sub(add(c, u), v), add(add(c, u), v), add(sub(c, u), v)]
        self.tri(mat, (p[0], p[1], p[2]), (n, n, n))
        self.tri(mat, (p[0], p[2], p[3]), (n, n, n))

    def box(self, mat, lo, hi):
        """An axis-aligned box, normals outwards."""
        c = mul(add(lo, hi), 0.5)
        h = mul(sub(hi, lo), 0.5)
        X, Y, Z = (h[0], 0, 0), (0, h[1], 0), (0, 0, h[2])
        self.quad(mat, add(c, X), (0, 0, -h[2]), (0, h[1], 0))   # +x
        self.quad(mat, sub(c, X), (0, 0, h[2]), (0, h[1], 0))    # -x
        self.quad(mat, add(c, Y), (h[0], 0, 0), (0, 0, -h[2]))   # +y
        self.quad(mat, sub(c, Y), (h[0], 0, 0), (0, 0, h[2]))    # -y
        self.quad(mat, add(c, Z), (h[0], 0, 0), (0, h[1], 0))    # +z
        self.quad(mat, sub(c, Z), (-h[0], 0, 0), (0, h[1], 0))   # -z

    def sphere(self, mat, c, r, n_lat=48, n_lon=96):
        """A UV sphere with smooth, outward normals."""
        def at(i, j):
            th = math.pi * i / n_lat
            ph = 2 * math.pi * j / n_lon
            d = (math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph))
            return add(c, mul(d, r)), d
        for i in range(n_lat):
            for j in range(n_lon):
                a, na = at(i, j)
                b, nb = at(i + 1, j)
                cc, nc = at(i + 1, j + 1)
                d, nd = at(i, j + 1)
                # counter-clockwise seen from outside
                if i > 0:
                    self.tri(mat, (a, cc, b), (na, nc, nb))
                if i < n_lat - 1:
                    self.tri(mat, (a, d, cc), (na, nd, nc))

    def bumpy_sphere(self, mat, c, r, subdivisions, bump):
        """An icosphere whose radius is modulated by `bump(direction)`, with
        normals averaged over the displaced faces: many small triangles."""
        t = (1 + math.sqrt(5)) / 2
        verts = [norm(v) for v in ((-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
                                   (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                                   (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1))]
        faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
                 (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
                 (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
                 (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
        for _ in range(subdivisions):
            mid = {}

            def midpoint(a, b):
                key = (min(a, b), max(a, b))
                if key not in mid:
                    verts.append(norm(mul(add(verts[a], verts[b]), 0.5)))
                    mid[key] = len(verts) - 1
                return mid[key]
            new = []
            for a, b, cc in faces:
                ab, bc, ca = midpoint(a, b), midpoint(b, cc), midpoint(cc, a)
                new += [(a, ab, ca), (b, bc, ab), (cc, ca, bc), (ab, bc, ca)]
            faces = new
        pts = [add(c, mul(d, r * (1 + bump(d)))) for d in verts]
        nrm = [(0.0, 0.0, 0.0)] * len(pts)
        for a, b, cc in faces:
            fn = cross(sub(pts[b], pts[a]), sub(pts[cc], pts[a]))
            for k in (a, b, cc):
                nrm[k] = add(nrm[k], fn)
        nrm = [norm(n) for n in nrm]
        for a, b, cc in faces:
            self.tri(mat, (pts[a], pts[b], pts[cc]), (nrm[a], nrm[b], nrm[cc]))

    def wall_with_door(self, mat, a, b, height, door_w, door_h):
        """A vertical wall from a to b on the floor (y = 0), with a doorway
        of door_w x door_h in its middle; door_w = 0 for none."""
        ax = sub(b, a)
        length = math.sqrt(dot(ax, ax))
        d = mul(ax, 1.0 / length)
        up = (0.0, 1.0, 0.0)

        def panel(s0, s1, y0, y1):
            if s1 - s0 <= 1e-6 or y1 - y0 <= 1e-6:
                return
            c = add(add(a, mul(d, 0.5 * (s0 + s1))), mul(up, 0.5 * (y0 + y1)))
            self.quad(mat, c, mul(d, 0.5 * (s1 - s0)), mul(up, 0.5 * (y1 - y0)))
        if door_w <= 0:
            panel(0, length, 0, height)
            return
        s0, s1 = 0.5 * (length - door_w), 0.5 * (length + door_w)
        panel(0, s0, 0, height)
        panel(s1, length, 0, height)
        panel(s0, s1, door_h, height)

    def write_glb(self, path):
        bin_parts, views, accessors = [], [], []

        def blob(data, target):
            off = sum(len(x) for x in bin_parts)
            pad = (-len(data)) % 4
            bin_parts.append(data + b"\0" * pad)
            views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data), "target": target})
            return len(views) - 1

        meshes, nodes = [], []
        for mat in sorted(self.meshes):
            pos, nrm, idx = self.meshes[mat]
            vp = blob(struct.pack("<%df" % (3 * len(pos)), *[x for p in pos for x in p]), 34962)
            vn = blob(struct.pack("<%df" % (3 * len(nrm)), *[x for n in nrm for x in n]), 34962)
            vi = blob(struct.pack("<%dI" % len(idx), *idx), 34963)
            lo = [min(p[k] for p in pos) for k in range(3)]
            hi = [max(p[k] for p in pos) for k in range(3)]
            accessors.append({"bufferView": vp, "componentType": 5126, "count": len(pos),
                              "type": "VEC3", "min": lo, "max": hi})
            accessors.append({"bufferView": vn, "componentType": 5126, "count": len(nrm),
                              "type": "VEC3"})
            accessors.append({"bufferView": vi, "componentType": 5125, "count": len(idx),
                              "type": "SCALAR"})
            a = len(accessors)
            meshes.append({"name": self.materials[mat][0],
                           "primitives": [{"attributes": {"POSITION": a - 3, "NORMAL": a - 2},
                                           "indices": a - 1, "material": mat}]})
            nodes.append({"name": self.materials[mat][0], "mesh": len(meshes) - 1})

        materials = []
        for name, albedo, emission in self.materials:
            m = {"name": name, "doubleSided": True,
                 "pbrMetallicRoughness": {"baseColorFactor": list(albedo) + [1.0],
                                          "metallicFactor": 0.0, "roughnessFactor": 1.0}}
            if any(e > 0 for e in emission):
                m["emissiveFactor"] = list(emission)
            materials.append(m)

        binary = b"".join(bin_parts)
        doc = {"asset": {"version": "2.0", "generator": "scenes/make_scenes.py"},
               "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}],
               "nodes": nodes, "meshes": meshes, "materials": materials,
               "accessors": accessors, "bufferViews": views,
               "buffers": [{"byteLength": len(binary)}]}
        js = json.dumps(doc, separators=(",", ":")).encode()
        js += b" " * ((-len(js)) % 4)
        total = 12 + 8 + len(js) + 8 + len(binary)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, total))
            f.write(struct.pack("<II", len(js), 0x4E4F534A))
            f.write(js)
            f.write(struct.pack("<II", len(binary), 0x004E4942))
            f.write(binary)
        ntri = sum(len(m[2]) // 3 for m in self.meshes.values())
        return ntri


def write_camera(path, eye, target):
    d = norm(sub(target, eye))
    yaw = math.degrees(math.atan2(d[2], d[0]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, d[1]))))
    with open(path, "w") as f:
        f.write("%.4f %.4f %.4f %.4f %.4f\n" % (eye[0], eye[1], eye[2], yaw, pitch))


# --------------------------------------------------------------------------
# the scenes

def stadium():
    s = Scene()
    grass = (s.material("grass_light", (0.30, 0.48, 0.22)), s.material("grass_dark", (0.20, 0.36, 0.16)))
    stands = s.material("stands", (0.55, 0.50, 0.45))
    concrete = s.material("pedestal", (0.6, 0.6, 0.6))
    statue = s.material("statue", (0.75, 0.62, 0.45))
    pole = s.material("pole", (0.4, 0.4, 0.42))
    s.material("mirror_ball")
    flood = s.material("floodlight", (0.0, 0.0, 0.0), (900.0, 880.0, 800.0))

    # The field: 12 x 12 tiles of 20 units, a checker of two greens.
    half, tile = 120.0, 20.0
    n = int(2 * half / tile)
    for i in range(n):
        for j in range(n):
            c = (-half + (i + 0.5) * tile, 0.0, -half + (j + 0.5) * tile)
            s.quad(grass[(i + j) % 2], c, (0, 0, tile / 2), (tile / 2, 0, 0))

    # Tiered stands on all four sides: four steps, each 6 high and 6 deep.
    for k in range(4):
        h0, h1 = 6.0 * k, 6.0 * (k + 1)
        d0, d1 = half + 6.0 * k, half + 6.0 * (k + 1)
        s.box(stands, (-d1, h0, d0), (d1, h1, d1))      # north / south
        s.box(stands, (-d1, h0, -d1), (d1, h1, -d0))
        s.box(stands, (d0, h0, -d0), (d1, h1, d0))      # east / west
        s.box(stands, (-d1, h0, -d0), (-d0, h1, d0))
    # The arena is closed — a wall above the top tier and a roof — so that the
    # light bounces around in it instead of escaping into a black sky after
    # one bounce, which would leave the photon map nearly empty.
    roof = s.material("roof", (0.55, 0.55, 0.58))
    outer, top = half + 24.0, 60.0
    s.quad(roof, (0.0, top, 0.0), (outer, 0, 0), (0, 0, outer))
    mid = 0.5 * (24.0 + top)
    hh = 0.5 * (top - 24.0)
    s.quad(roof, (0.0, mid, -outer), (outer, 0, 0), (0, hh, 0))
    s.quad(roof, (0.0, mid, outer), (-outer, 0, 0), (0, hh, 0))
    s.quad(roof, (outer, mid, 0.0), (0, 0, outer), (0, hh, 0))
    s.quad(roof, (-outer, mid, 0.0), (0, 0, -outer), (0, hh, 0))

    # The camera stands near the south end, looking north across the field.
    eye = (0.0, 1.7, 100.0)
    # A detailed object 2.5 units ahead: a bumpy sphere on a pedestal.
    s.box(concrete, (0.15, 0.0, 97.2), (1.05, 0.7, 98.1))
    s.bumpy_sphere(statue, (0.6, 1.2, 97.65), 0.45, 6,
                   lambda d: 0.12 * math.sin(7 * d[0]) * math.sin(6 * d[1]) * math.sin(8 * d[2]))
    # A mirror ball 30 units away.
    s.sphere(s.material("mirror_ball"), (-9.0, 5.0, 70.0), 5.0)

    # Four floodlights on poles at the corners, 45 up, facing the centre.
    for sx in (-1, 1):
        for sz in (-1, 1):
            base = (sx * 108.0, 0.0, sz * 108.0)
            s.box(pole, (base[0] - 0.6, 0.0, base[2] - 0.6), (base[0] + 0.6, 44.0, base[2] + 0.6))
            c = (base[0], 45.0, base[2])
            towards = norm(sub((0.0, 0.0, 0.0), c))
            u = mul(norm(cross(towards, (0.0, 1.0, 0.0))), 5.0)
            v = mul(norm(cross(towards, u)), 2.5)
            # normal u x v = u x (t x u) = t |u|^2: the light faces the field
            s.quad(flood, add(c, mul(towards, 0.7)), u, v)
    return s, eye, (0.0, 1.0, 0.0)


def rooms():
    s = Scene()
    wall = s.material("wall", (0.72, 0.70, 0.66))
    floor = s.material("floor_wood", (0.50, 0.36, 0.22))
    ceiling = s.material("ceiling", (0.80, 0.80, 0.80))
    furniture = s.material("furniture", (0.35, 0.45, 0.60))
    lamp = s.material("ceiling_lamp", (0.0, 0.0, 0.0), (30.0, 28.0, 24.0))
    s.material("mirror_panel")

    room, height, n = 8.0, 3.0, 3
    size = room * n
    s.quad(floor, (size / 2, 0.0, size / 2), (0, 0, size / 2), (size / 2, 0, 0))          # up
    s.quad(ceiling, (size / 2, height, size / 2), (size / 2, 0, 0), (0, 0, size / 2))     # down
    # Outer walls, no doors.
    s.wall_with_door(wall, (0, 0, 0), (size, 0, 0), height, 0, 0)
    s.wall_with_door(wall, (0, 0, size), (size, 0, size), height, 0, 0)
    s.wall_with_door(wall, (0, 0, 0), (0, 0, size), height, 0, 0)
    s.wall_with_door(wall, (size, 0, 0), (size, 0, size), height, 0, 0)
    # Inner walls, one segment per pair of neighbouring rooms; most have a
    # doorway, three are closed so that some rooms are reached only the long way.
    closed = {("x", 1, 1), ("z", 2, 0), ("x", 2, 2)}
    for k in (1, 2):
        for j in range(n):
            door = 0 if ("x", k, j) in closed else 1.2
            s.wall_with_door(wall, (room * k, 0, room * j), (room * k, 0, room * (j + 1)),
                             height, door, 2.2)
            door = 0 if ("z", k, j) in closed else 1.2
            s.wall_with_door(wall, (room * j, 0, room * k), (room * (j + 1), 0, room * k),
                             height, door, 2.2)
    # Three lit rooms: a lamp panel just under the ceiling, facing down.
    for (i, j) in ((0, 0), (2, 1), (1, 2)):
        c = (room * (i + 0.5), height - 0.02, room * (j + 0.5))
        s.quad(lamp, c, (0.6, 0, 0), (0, 0, 0.6))  # (0.6,0,0) x (0,0,0.6) = -y
    # Furniture in every room, and a mirror panel on one wall of the first.
    for i in range(n):
        for j in range(n):
            x0, z0 = room * i, room * j
            s.box(furniture, (x0 + 1.0 + i, 0.0, z0 + 5.2), (x0 + 2.6 + i, 0.8, z0 + 6.4))
            s.box(furniture, (x0 + 5.6, 0.0, z0 + 1.0 + j), (x0 + 6.6, 1.9, z0 + 1.6 + j))
    s.quad(s.material("mirror_panel"), (0.02, 1.4, 3.0), (0, 0, -1.0), (0, 0.9, 0))  # faces +x
    eye = (1.6, 1.6, 1.4)
    return s, eye, (9.0, 1.1, 4.0)


def caustic():
    s = Scene()
    floor = s.material("floor", (0.75, 0.75, 0.75))
    walls = s.material("walls", (0.65, 0.62, 0.58))
    s.material("glass_ball")
    light = s.material("small_light", (0.0, 0.0, 0.0), (400.0, 390.0, 360.0))

    half = 6.0
    s.quad(floor, (0, 0, 0), (0, 0, half), (half, 0, 0))                    # up
    s.quad(walls, (-half, 3.0, 0), (0, 0, -half), (0, 3.0, 0))              # +x
    s.quad(walls, (0, 3.0, -half), (half, 0, 0), (0, 3.0, 0))               # +z
    s.sphere(s.material("glass_ball"), (0.0, 1.0, 0.0), 1.0, 64, 128)
    # A 0.5 x 0.5 light up and behind the ball, facing down; low enough that
    # the ball focuses it onto the floor beside itself, inside its shadow.
    s.quad(light, (-3.2, 3.2, -2.4), (0.25, 0, 0), (0, 0, 0.25))           # -y
    eye = (4.6, 2.6, 4.2)
    return s, eye, (0.2, 0.5, 0.3)


def main():
    for name, make in (("stadium", stadium), ("rooms", rooms), ("caustic", caustic)):
        scene, eye, target = make()
        ntri = scene.write_glb(os.path.join(HERE, name + ".glb"))
        write_camera(os.path.join(HERE, name + "_camera.txt"), eye, target)
        print("%-8s %7d triangles, %d materials -> scenes/%s.glb, scenes/%s_camera.txt"
              % (name, ntri, len(scene.materials), name, name))


if __name__ == "__main__":
    main()
