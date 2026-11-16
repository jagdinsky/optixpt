# Photon gather: three method families

Progressive photon mapping (Knaus–Zwicker schedule, α = 2/3) on the GPU, comparing
where the search structure sits and what it holds:

- **A — gather:** photons go into a structure; each visible point searches it.
- **B — gather with culling:** the same, but only photons some visible point can reach are stored.
- **C — splatting:** the visible points go into a structure; each photon, as it is deposited, adds its flux to the visible points it lies within. No photons are stored.

All numbers below come from one build (Release, OptiX validation off, RTX 4060 Laptop GPU, driver 595.99.02, CUDA 12.8, OptiX 9.1). Configurations are named `<family>/<structure>/<radius mode>`.

<!-- SUMMARY -->

## 1. Taxonomy

| family | structure | the structure holds | the photon pass stores | result |
|---|---|---|---|---|
| A | brute | — | every photon | exact (the reference) |
| A | uniform | photon ids in a dense grid, cells = the frame's largest r | every photon | exact |
| A | hash | photon ids per cell in a hash table, cells = the frame's largest r | every photon | exact |
| A | stochastic | one copy of the last photon written per bucket | every photon | **biased** (legacy baseline) |
| A | rect | one photon per bucket, the largest random key | every photon | unbiased |
| A | mlhash | photon ids per (cell, level), every level in use | every photon | exact |
| A | mlrect | one entry per bucket over (cell, level), the smallest random key | every photon | unbiased |
| B | same as A | the same, over the culled map | photons in a cell some lookup marked | as in A |
| C | splat | visible points per cell, cells = F × the frame's largest r | nothing | exact |
| C | mlsplat | visible points per (cell, level), each point on the level of F × its r | nothing | exact |
| C | stochsplat | mlsplat's grid; each deposit adds m·w·Φ to one of the m points it reaches | nothing | unbiased |

Every exact configuration finds exactly the same (photon, visible point) pairs as brute force for the same seed, frame and radius mode; they differ only in the order of float sums. Every structure visits the cells the lookup's sphere overlaps (8–27 cells between r and 2r wide), culling marks exactly those cells at the lookup's level, and splatting files a visible point under exactly those cells.

## 2. What was built

**Measurement.**
- **`--validation on|off`:** default off.
- **Build directory:** measurements use `build-release/`. The editor's CMake integration reconfigures `build/` to Debug whenever `CMakeLists.txt` changes, and did so during this work.
- **Timing JSON:** every file records the build type, validation, GPU and driver.
- **Query passes:** CUDA kernels, one per structure and per `--stats` setting, generated from one template (`src/gather.h`). CUDA kernels were chosen over one OptiX pipeline per query for three reasons:
  - each kernel gets its own register allocation by construction;
  - registers can be read at runtime (`cudaFuncGetAttributes`, in every timing JSON);
  - no pipeline or SBT is needed per variant.
- **OptiX pipelines:** a "trace" pipeline (path tracer, camera, photon, lightvis), one for the splatting photon pass, one for stochastic splatting, and one per fused structure.
- **Culling marks:** moved out of the camera program into a timed CUDA mark pass.

**Shared definitions.**
- **One cell-range function (`cellRange`)** is used by every query, the culling marks and the splatting grid.
- **Culling-mask hash:** an extra `fmix32` makes it independent of the table buckets.
- **Table sizing:** every hash table has T = max(1024, nextPow2(4 · entries)) buckets (`--table-load`). Two exceptions: tables filled during the photon pass count the photon capacity as their entries, and with culling mlrect counts its insertions first.
- **Bit-identical pair test:** the pair test, the cone weight and each pair's product w·Φ use explicitly rounded arithmetic (`fmaf`, `__fmul_rn`, …), which no compiler may fuse. The OptiX programs and the CUDA kernels therefore compute bit-identical distances, weights and contributions. That is what lets splatting (OptiX) match the gathers (CUDA) pair for pair.

**Family C.**
- `--accel splat | mlsplat | stochsplat`.
- `--splat-cell-factor 1|2` and `--splat-stripes K`.
- The visible-point grid build: count, scan, emit 64-bit keys, stable radix sort on the bucket bits, `DeviceSelect::Unique`, then cell offsets.
- Finalize, and `--verify-gather` for family C (each frame against hash/mlhash over the same gather points and photon paths).

**Optional (all off by default).**
- `--inline-positions`: photon positions next to the hash/mlhash entries, and point positions and radii next to the splatting entries.
- `--hier-mask`: parent-cell mask, tested coarse to fine.
- `--fused`: family A, camera walk and lookup in one program, with the structure built from the previous frame's radii and levels.

**Tooling.**
- **Renderer flags:** `--checkpoints [LIST]` and `--dump-frames LIST`.
- **`eval/`:** `render_references.py` (path-traced references, which record their own noise), `error_vs_time.py` (relMSE against GPU time; CSV, SVG, slopes, best k_pix) and `variance.py` (the E7 diagnostic).
- **Scenes:** `scenes/make_scenes.py` writes the stadium, rooms and caustic scenes.
- **Drivers:** `checks.py v1..v7` and `exec.py --experiments [--reduced]`.

## 3. Checks

All seven pass on the final build.

| check | what | result |
|---|---|---|
| V1 | every existing configuration renders the image it rendered before (old build frozen in `build/pre_families`) | **PASS** — 16 exact configurations (uniform, hash, mlhash, brute × global/footprint × culling off/on) within 3.8e-7 relative of the old build (run-to-run floor 2.4e-7); rect within 2.8e-6 (allclose 1e-5/1e-8); stochastic and mlrect, which are not reproducible run to run in either build, differ by their run-to-run floor |
| V2 | range-based visits: same images, cells and candidates per lookup | **PASS** — identical accepted photons per lookup for every reproducible structure. Footprint: 5.6 cells instead of 27; uniform 168 → 58 candidates, hash 178 → 59; mlhash 14.6 cells, 72 → 38 candidates (global 501 → 375). Uniform, hash and mlhash = brute force at frames 1, 100 and 1000 in both modes (≤ 5.9e-7) |
| V3 | mask decoupling | **PASS** — culled = unculled per frame for brute, uniform, hash and mlhash in both modes (≤ 3.1e-7; 63–73% of deposits culled). B/mlhash tested entries per lookup: before 154 vs 135 without culling (+14%); after 73.3 vs 74.2 (−1%). rect and mlrect converge to hash and mlhash like M^−0.48…−0.50 with and without culling |
| V4 | splat = hash and mlsplat = mlhash per frame | **PASS** — frames 1, 100, 1000; global and footprint; F = 1 and 2; K = 1 and 4: worst max rel 2.9e-6, no pixel off |
| V5 | stochastic structures unbiased (`--no-ppm`, footprint radius) | **PASS** — rect→hash M^−0.48 (L2) / −0.50 (L1), mlrect→mlhash M^−0.49 / −0.50, stochsplat→mlsplat M^−0.49 / −0.49 |
| V6 | fused A = split A | **PASS** — uniform, hash, mlhash, global and footprint: ≤ 4.0e-7 (floor 2.5e-7) |
| V7 | relMSE against the path-traced reference falls at −2/3 (Cornell box) | **PASS** — A/mlhash/footprint −0.647, B/mlhash/footprint −0.677, A/hash/global with r₁ = 0.5% −0.636. At the default r₁ = 1.5% the global run falls at −0.485: with large early radii its error is still bias-dominated after 1024 frames |

### V1: per-pass times and registers

1280×720, 1M photon paths per frame, depth 8, 128 frames (8 warm-up), median of 3 runs, ms per frame. "Before" is the old build: OptiX validation always on, one pipeline holding every query, 3×3×3 blocks, 2 buckets per entry, culling marks inside the camera pass.

| configuration | total before | new, validation on, load 2 | new, validation off, load 2 | new defaults (load 4) | query before → after |
|---|---|---|---|---|---|
| uniform/global | 24.37 | 24.17 | 23.58 | 23.60 | 18.15 → 17.96 |
| hash/global | 28.34 | 28.26 | 27.69 | 28.10 | 21.83 → 21.72 |
| hash/footprint | 20.27 | 13.16 | 12.58 | 13.29 | 13.33 → 6.16 |
| mlhash/footprint | 28.60 | 26.57 | 26.02 | 28.40 | 9.33 → 7.28 |
| B/mlhash/footprint | 17.33 | 12.18 | 11.56 | 11.90 | 10.68 → 5.58 |
| rect/global | 10.68 | 8.30 | 8.01 | 8.99 | 3.66 → 1.15 |
| rect/footprint | 12.23 | 9.26 | 9.08 | 10.11 | 3.61 → 0.60 |
| mlrect/footprint | 27.42 | 29.46 | 28.84 | 31.04 | 1.83 → 1.95 |
| B/mlrect/footprint | 9.24 | 8.57 | 7.99 | 8.61 | 1.80 → 1.42 |

- **OptiX validation** cost about 0.5 ms per frame on the photon pass and 0.1 ms on the camera pass; the CUDA queries are unaffected by it.
- **Range visits** cut the single-level footprint queries in half, because a small lookup no longer walks 27 cells sized for the largest radius.
- **Per-structure kernels** cut rect's query by 3–6×. It no longer shares a raygen and its register budget with every other structure, and it loads a representative's fields only when needed.
- **mlrect without culling got slower.** At the same nominal load its table now counts every insertion (2^25 instead of 2^24 buckets), and the build pays for the larger table.
- **Table load 4 (the specified default) costs build time:** +0.7 ms (hash), +2.4 ms (mlhash), +0.8 ms (rect, the clear), +2.2 ms (mlrect) per frame, for at most 0.3 ms of query time.

Registers per thread of the query kernels (the OptiX programs need `sudo ncu`; profiling counters are not accessible here):

| kernel | brute | uniform | hash | stochastic | rect | mlhash | mlrect | mark |
|---|---|---|---|---|---|---|---|---|
| registers | 39 | 48 | 48 | 54 | 53 | 55 | 55 | 31 |
| local bytes | 0 | 0 | 256 | 256 | 0 | 0 | 0 | 0 |

(The 256 local bytes are the visited-bucket list of the single-level hash walk.)

<!-- EXPERIMENTS -->

## 6. Caveats and open items

- **Reduced matrix.** Error against time (E5, E6) uses 1 seed and 1024 frames; E7 uses 256 frames with 16 dumped frames per run. The full matrix (3 seeds, 4096 frames) is `RENDERER_BUILD=build/experiments/bin python3 exec.py --experiments`.
- **References.** Their own relMSE: Cornell 3.0e-5 (16,384 samples per pixel), rooms 8.9e-5 (16,384), stadium 7.2e-5 (65,536), caustic 1.9e-4 (524,288). Curves that approach these levels measure the reference, not the method; the figures draw them as a floor.
- **Hierarchical mask.** It is exact but not the stored-photon set of the flat mask. It also drops photons that only a hash collision on a finer level keeps, so with it rect/mlrect give a different (still unbiased) realization.
- **Not tested.** The realtime GUI (new accel entries, splatting path) was compiled but not run interactively. Register counts of the OptiX programs (camera, photon, splatting) need `sudo ncu --metrics launch__registers_per_thread`.
- **Not committed.** Nothing is committed. The frozen "before" sources are in `build/pre_families/`, the experiment binary in `build/experiments/bin/`.
