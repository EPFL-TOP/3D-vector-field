# CoTracker3 for pseudo-3D registration — investigation findings

Dataset: `Defective_somite.tif` — zebrafish somitogenesis 4D stack.
Env: `microscope-tracking` (Python 3.13.5, torch 2.7.1 + MPS, SimpleITK 2.5.5,
itk-elastix 0.25.3, opencv, scikit-image, tifffile). CoTracker3 cached at
`~/.cache/torch/hub/facebookresearch_co-tracker_main` (offline/online checkpoints present).

## TL;DR verdict
- **Do NOT use CoTracker3 as the registration engine, and do NOT use its confidence to
  match z-slices** — that step is a category error (confidence = the tracker's belief its
  own 2D prediction is within ~12 px of truth, not a cross-image/cross-slice similarity).
  Empirically CoTracker3 loses ~all tracked points within ~10–15 frames on this data.
- **A single 2D z-plane cannot capture the motion**: tissue moves *through z* and deforms,
  so per-slice 2D registration is fundamentally limited.
- **Use full-3D intensity-based registration** (ITK-Elastix presets, or SimpleITK with the
  **NCC/Correlation** metric — *not* Mattes-MI), estimated on the best channel and applied to
  all channels. Provide **rigid/affine (stabilized, development preserved)** and
  **deformable B-spline (development absorbed)** outputs and compare them.
- CoTracker3 is at most an optional short-window 2D **drift diagnostic**.

## Verified data facts
| property | value | implication |
|---|---|---|
| shape (TZCYX) | 179 × 71 × 3 × 303 × 303, uint16 | small XY, deep Z, long T |
| voxel size | **xy = 0.347 µm**, z = 1.5 µm (anisotropy ≈ 4.3×) | set sitk spacing (0.347,0.347,1.5) |
| best channel | **ch0** (Laplacian-var 693k vs 119k/164k) — dense nuclei-like puncta | register on ch0, apply to all |
| ch1 / ch2 | near-empty / sparse | not useful as primary metric |
| best-focus z | ~33 | — |
| photobleaching | ch0 mean ~halves by t≈90 | NCC on per-frame-normalized data; not SSD |
| in-plane drift | ~5 px/step mean, up to 70 px/step, **+224 px** cumulative in Y (≈78 µm) | large; needs robust/sequential reg |
| z-drift (NCC-argmax offset vs t0) | 0 at t1/t5, +1 at t40, **−9 planes at t90, −11 (16.5 µm) at t178** | through-plane motion is large *cumulatively* → to-reference degrades late; use sequential |
| z-match similarity (mean max-NCC) | 0.71 (t1) → 0.52 (t5) → 0.34 (t90) → 0.22 (t178) | NCC-over-z is the right z signal (not CoTracker) |

## CoTracker3 investigation (empirical)
- Runs on **MPS** ✓. Entrypoints `cotracker3_offline` (full-sequence, window 60) /
  `cotracker3_online` (causal, window 16). Input `(B,T,3,H,W)` float **0–255**, 3 channels
  required (replicate grayscale). Output `tracks (B,T,N,2)`, `visibility (B,T,N)`.
- **The model returns `(coords, visibility, confidence, …)` — a graded [0,1] confidence
  exists** (heads + sigmoid), but the public predictors hide it: offline thresholds
  `visibility > 0.9` and discards confidence; online does `visibility*confidence > 0.6`.
  You only get the soft score by calling `model.forward` directly / monkeypatching
  (done in `cotracker_register_demo.py`).
- Confidence semantics (paper + `losses.py` `sequence_prob_loss`): BCE target = indicator
  that the predicted track is within **12 px** of ground truth *in the current frame*. The
  vis/confidence heads were **frozen/unsupervised** in the real-video training stage → weak
  calibration on OOD fluorescence.
- **Empirical result on ch0 z33** (frames 0–63, 30×30 grid): visibility decays 1.0→~0 by
  frame ~15; the grid **collapses into a low-confidence blob** by frame 32 (see
  `out/tracks_overlay.png`). Confidence honestly flags the failure (good calibration of the
  *failure*, not usable tracks). Track-based XY registration ran out of points (median 0/900).
- → CoTracker3 is the wrong tool for this registration; the failure is intrinsic
  (OOD fluorescence + large motion + bleaching + repetitive structure).

## 3D registration backbone (empirical, verified)
Metric matters: **Mattes-MI + GradientDescent diverged** on this low-texture same-modality
data (t0→t10 rigid 0.24, affine −0.07). Switching to **Correlation (NCC) +
RegularStepGradientDescent** (multi-res [4,2,1], scores in valid overlap mask only) fixes it:

| pair | raw@overlap | rigid (NCC) | recovered (x,y,z) px |
|---|---|---|---|
| t0→t10 | 0.589 | **0.666** | (−38, +21, 0.3) — matches phase-corr drift |
| t0→t40 | 0.584 | **0.671** | (−10, +46, 9.6) — note z drift |

Deformable B-spline (affine-initialized, single-res mesh) further improves NCC
(measured ~0.50→0.66 on a hard pair) but is slow.

### Pitfalls found
- Scoring NCC over zero-padded resample borders is misleading → mask to valid overlap, fill
  with background (not 0) when chaining stages.
- **Direct large-gap (to-t0) registration is brittle** (t10 diverged with MI) → register
  **sequentially t→t+1** and compose, or use a temporal-median / piecewise reference.
- Multi-res B-spline `scaleFactors` triggers an LBFGSB scales-vs-parameters error → use
  single-resolution mesh (or per-level mesh) + bending-energy regularization.
- Naive `skimage.phase_cross_correlation` 3D gave a **false z-peak** → don't use as engine.

## Recommended architecture
1. **Preprocess** (per frame): percentile-normalize ch0 (1–99.5), optionally histogram-match
   to the reference, to counter bleaching. Set sitk spacing **(0.347, 0.347, 1.5)** on every channel.
2. **Stabilization (goal a)** — estimate on **ch0**, apply identical transform to all 3 channels:
   - Optional fast 2D pre-align: **pystackreg** (StackReg) on a max-z-projection / best-focus plane.
   - 3D **rigid (Euler3D) → affine**, **NCC metric**, RegularStepGradientDescent, pyramid [4,2,1].
   - Engine: **ITK-Elastix default parameter maps** (already installed; lowest-tuning-risk) or the
     verified SimpleITK config in `register_3d_demo.py` / `verify_ncc_metric.py`.
   - **Reference strategy: sequential t→t+1, compose** (robust to the −11-plane cumulative z-drift);
     fall back to to-reference only for short spans.
3. **Deformable (goal b)** — B-spline (ITK-Elastix bspline map or SimpleITK BSpline), affine-initialized,
   NCC metric, **bending-energy regularization**, **Jacobian > 0 (no-folding) check** per frame.
4. **Compare a vs b** — keep **affine-only as the primary stabilized output** (preserves real
   somite growth); treat the **B-spline residual displacement / Jacobian field as the quantified
   developmental deformation**, not as registration error.
5. **Build the registered 4D image** — apply composed transforms, resample all channels, stream with
   `tifffile.memmap` (never load the whole 7 GB), write OME-TIFF.

### Runtime budget (CPU, measured)
rigid ≈ 18–30 s/pair, single-res B-spline ≈ 95 s/pair. Full 179-frame affine→B-spline = **hours**
→ downsample / coarse-to-fine / cap iterations / run in background.

## Tool inventory
- Installed & verified: SimpleITK 2.5.5, **itk-elastix 0.25.3**, torch 2.7.1+MPS, opencv,
  scikit-image, tifffile, CoTracker3 (torch hub cache).
- Not installed (optional): **pystackreg** (trivial, user-named, recommended baseline),
  antspyx 0.6.3 (cp313 arm64 wheel exists → diffeomorphic SyN cross-check), VoxelMorph/MONAI
  (brain-MRI pretrained weights won't transfer; needs self-training).

## Scripts (in `investigation/`)
- `inspect_data.py` — data characterization (channels, focus, drift, z-match NCC).
- `smoke_cotracker.py` — CoTracker3 + MPS smoke test.
- `cotracker_register_demo.py` — CoTracker3 on real data + soft-confidence extraction + XY reg attempt.
- `register_3d_demo.py` — staged 3D rigid→affine→B-spline (masked NCC).
- `verify_ncc_metric.py` — confirms NCC-metric fix for the divergent pair.
- `raw_inspect.py` — characterize RAW unregistered timepoints around the contraction.
- `prototype_field.py` — STAGE-1 dense displacement-field estimator (rigid + diffeomorphic Demons).
- `out/` — all PNGs + summary JSONs.

## UPDATE — real raw data + reframed goal (dense displacement field)
The earlier `Defective_somite.tif` was a CROPPED, ALREADY-REGISTERED volume → its motion numbers
were residual artifacts. Real goal (clarified): NOT classic register-to-template, but compute a
**dense 3D inter-frame displacement field** u(x) per consecutive pair, then compose fields so a
selected material region "stays fixed" when scrolling time (Lagrangian / co-moving frame). The
field gradient also quantifies the contraction (strain). CoTracker is fully out (a 2D point
tracker can't survive the contraction; correspondence breaks).

Raw data: `tNNNN_Channel 1.tif` (channel 1 = nuclear marker), each Z=215 × 2304 × 2304 uint16,
COMPRESSED (not memory-mappable → read via TiffFile pages). Resolution tag is a 96-dpi
placeholder → **xy calibration unknown** (working assumption xy≈0.347µm, z=1.5µm → ~4.3× aniso;
CONFIRM with user). Heat-shock contraction = tail curling; baseline stable (t01↔t14 NCC 0.89),
contraction at **t15→t16** (~40px XY + ~37 z-planes gross motion, NCC 0.99→0.76), then ongoing
non-rigid change t16→t18 with a sharp signal drop (p99 3400→1080).

STAGE-1 PROTOTYPE RESULT (t15→t16, hardest pair, XY//4 full-Z, rigid+Demons, hist-matched):
NCC valid-overlap **0.748 → rigid 0.939 → +Demons 0.986**; field is **diffeomorphic**
(Jacobian min 0.061, **0% folding**) → composable. Runtime ~11 min/pair (loading + 4-level rigid +
80 Demons iters) → needs optimization (volume caching, leaner pyramid, more downsampling, or
torch/MPS optical flow) before running all ~179 frames. Approach is FEASIBLE.

## Cell-tracking plan (segmentation + linking with the field as motion prior)
Goal: track nuclei before/through/after the contraction. CoTracker stays OUT.
Pipeline: (1) 3D nuclei segmentation per frame — **Cellpose v4/SAM** (PyTorch, do_3D, anisotropy
≈4.3) as zero-shot baseline; StarDist-3D (needs training, TF) as accuracy upgrade. (2) Use the
SimpleITK displacement field as the **motion prior**: warp each nucleus centroid by the field to
predict t+1, then LAP/KDTree link the small residual. (3) Linker = **Ultrack** (`add_flow()`
ingests an external dense field and applies it to candidates before linking). Build a ~100-line
scipy `linear_sum_assignment` + `cKDTree` warp-then-link baseline FIRST.

### CRITICAL integration caveats (red-team verified — fix before trusting any track)
1. **Units**: the Demons/field values are in **physical microns** (NRRD space dirs in µm; spacing
   (1.388,1.388,1.5)). Ultrack `add_flow` wants displacement as a FRACTION of axis extent in
   VOXELS. Convert µm→voxels FIRST (dz/1.5, dy/1.388, dx/1.388), THEN /Z,/Y,/X. Dividing µm by
   voxel-count is wrong and makes z ~4.3× off while still passing the |v|≤1 assert (silent bug).
2. **Compose rigid+Demons**: the Demons residual alone OMITS the bulk rigid motion (e.g. t15→t16
   z=−55.6µm; t17→t18 y=89.3µm). Must compose rigid∘Demons into ONE field per pair
   (`TransformToDisplacementField`) before use — currently only separate `.tfm`+`.nrrd` are on disk.
3. **Direction**: fields map fixed(t)→moving(t+1). For "where does a cell at t go at t+1" you likely
   need the INVERSE (fields are diffeomorphic/invertible). Confirm sign on a landmark.
4. **Hardest pair is t17→t18** (y=89.3µm, worst post-Demons NCC 0.907), NOT t15→t16. Validate
   FLOW-vs-NO_FLOW linking on BOTH.
5. **Storage**: fields are currently double-precision NRRD (~1.65GB each). Convert to **float16 zarr**,
   written incrementally per timepoint (a full (T,3,Z,Y,X) float16 array ≈76GB — never materialize it).
6. **Resolution scaling**: field is XY//4 full-Z; Cellpose labels may be at a different res. Make flow
   and label array dims a consistent integer ratio per axis; unit-test by warping one known centroid.

## Windows production environment (verified)
Four isolated conda envs (never mix TF + torch CUDA): `environment-core.yml` (registration, CPU),
`environment-segtrack.yml` (Cellpose+Ultrack, torch installed separately from PyTorch CUDA index),
`environment-cotracker.yml` (optional, isolated), `environment-stardist-tf210-winGPU.yml` (frozen
last-resort; prefer Cellpose or StarDist-under-WSL2). Key Windows facts: TF native-Windows GPU
ended at 2.10 → use Cellpose(torch); torch wheels bundle CUDA runtime (only NVIDIA driver needed);
set LongPathsEnabled, install MSVC redistributable, conda-forge only. Segmentation wants a GPU
(CPU 3D ≈ hours/volume); registration + Ultrack link/solve are fine on CPU.
</content>
