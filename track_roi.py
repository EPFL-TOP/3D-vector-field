#!/usr/bin/env python
"""
Follow a material ROI across the movie using the precomputed displacement fields, so the
selected region stays FIXED while the rest of the embryo moves. Reuses the fields/ and the
zarr cache produced by compute_fields_movie.py.

  For each frame k: compose the anchor->k field incrementally (evaluated on the small ROI
  grid -> cheap), then resample frame k into the ROI grid. The ROI box is defined once in the
  anchor frame; the field carries it through the contraction.

Tips for a GOOD-looking result (the failure mode is the region leaving the imaged volume or
accumulation over a very long follow):
  * Follow a SHORT, relevant window with --span (e.g. 30-40 frames around the event), not all 180.
  * Put the ROI on a CENTRAL region (--center "x,y,z" in microns) that stays in the field of view.
  * Use --out-xy-ds 2 for cell-resolution crops (loads finer volumes; slower IO).
  * Check roi_trajectory.png: if "fraction inside FOV" drops below ~1, the region is leaving the
    imaged volume -> choose a different center/window.

Outputs (in --out):
  roi_comoving.tif  (T,Z,Y,X)  the ROI followed (should look ~still when scrolling T)
  roi_rawbox.tif    (T,Z,Y,X)  the SAME fixed image-box, NOT followed (material drifts out)
  roi_pmip.gif/.mp4            pseudo z-MIP per frame: [raw fixed box | co-moving]
  roi_trajectory.png/.csv      QC: fraction of ROI inside FOV + center displacement (um) vs frame

Example (server, 40-frame window around the contraction, cell-resolution):
  python track_roi.py --input "H:\\...\\t*_Channel 1.tif" --fields-dir D:\\out\\fields ^
     --cache-dir D:\\cache --out D:\\out\\roi --xy-ds 4 --anchor 13 --span 40 --out-xy-ds 2 --size-um 60
"""
import argparse, glob, os, csv
import numpy as np
import tifffile
import SimpleITK as sitk
import imageio.v2 as imageio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compute_fields_movie import sitkimg, spacing, read_vol, load_vol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--fields-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--xy-ds", type=int, default=4, help="MUST match the field resolution")
    ap.add_argument("--anchor", type=int, default=0)
    ap.add_argument("--span", type=int, default=0, help="follow only anchor..anchor+span frames (0 = to end)")
    ap.add_argument("--center", default=None, help='"x,y,z" microns; default = auto (bright-tissue centroid)')
    ap.add_argument("--size-um", type=float, default=60.0)
    ap.add_argument("--out-xy-ds", type=int, default=0, help="crop render resolution (0 = same as --xy-ds; 2 = cell-level)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(args.input))
    cache = os.path.join(args.cache_dir, f"volcache_ds{args.xy_ds}.zarr") if args.cache_dir else None
    out_ds = args.out_xy_ds or args.xy_ds
    sp_f, sp_o = spacing(args.xy_ds), spacing(out_ds)      # field-res / crop-res spacings (x,y,z um)

    # --- choose ROI center (microns), from the anchor volume at field resolution ---
    va = read_vol(cache, files, args.anchor, args.xy_ds)
    if args.center:
        cphys = np.array([float(s) for s in args.center.split(",")])
    else:
        thr = np.percentile(va, 90); m = va >= thr
        idx = np.argwhere(m).astype(np.float64); w = va[m].astype(np.float64)
        cz, cy, cx = (idx * w[:, None]).sum(0) / w.sum()
        cphys = np.array([cx, cy, cz]) * np.array(sp_f)
    print(f"ROI center micron(x,y,z)={cphys.round(1)}  size={args.size_um}um  out_xy_ds={out_ds}")

    # --- ROI grid (anchor physical frame) at crop resolution ---
    nx, ny, nz = (int(round(args.size_um / sp_o[i])) for i in range(3))
    half = args.size_um / 2.0
    roi = sitk.Image(nx, ny, nz, sitk.sitkFloat32)
    roi.SetSpacing(sp_o); roi.SetOrigin(tuple(cphys - half))
    tff = sitk.TransformToDisplacementFieldFilter(); tff.SetReferenceImage(roi)

    def fld(i):
        return sitk.DisplacementFieldTransform(
            sitk.Cast(sitk.ReadImage(os.path.join(args.fields_dir, f"field_{i:04d}.nrrd")), sitk.sitkVectorFloat64))
    def getvol(k):
        return read_vol(cache, files, k, args.xy_ds) if out_ds == args.xy_ds else load_vol(files[k], out_ds)

    ident = sitk.Transform(3, sitk.sitkIdentity)
    end = len(files) if args.span <= 0 else min(len(files), args.anchor + args.span + 1)
    co, rawbox, inb, disp = [], [], [], []
    M = None
    for k in range(args.anchor, end):
        vnp = getvol(k); volk = sitkimg(vnp, out_ds); ones = sitkimg(np.ones_like(vnp), out_ds)
        if k == args.anchor:
            phi = ident; M = None
        else:
            P = fld(k-1)
            if M is None:
                phi = P
            else:
                c = sitk.CompositeTransform(3); c.AddTransform(P); c.AddTransform(M); phi = c
            M = sitk.DisplacementFieldTransform(tff.Execute(phi)); phi = M
        co.append(sitk.GetArrayFromImage(sitk.Resample(volk, roi, phi, sitk.sitkLinear, 0.0)))
        rawbox.append(sitk.GetArrayFromImage(sitk.Resample(volk, roi, ident, sitk.sitkLinear, 0.0)))
        inb.append(float(sitk.GetArrayFromImage(sitk.Resample(ones, roi, phi, sitk.sitkNearestNeighbor, 0.0)).mean()))
        mapped = np.array(phi.TransformPoint(tuple(cphys)))
        disp.append(float(np.linalg.norm(mapped - cphys)))
        print(f"  ROI frame {k:04d}  inFOV={inb[-1]:.2f}  centerMoved={disp[-1]:.1f}um")
    co = np.stack(co); rawbox = np.stack(rawbox)

    def u8(a, lo, hi): return (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    lo, hi = np.percentile(co, [1, 99.5])
    tifffile.imwrite(f"{args.out}/roi_comoving.tif", u8(co, lo, hi), imagej=True, metadata={"axes": "TZYX"})
    tifffile.imwrite(f"{args.out}/roi_rawbox.tif", u8(rawbox, lo, hi), imagej=True, metadata={"axes": "TZYX"})

    co_m, raw_m = co.max(1), rawbox.max(1)
    lo2, hi2 = np.percentile(co_m, [1, 99.5])
    sep = np.full((co_m.shape[1], 4), 255, np.uint8)
    frames = [np.hstack([u8(raw_m[t], lo2, hi2), sep, u8(co_m[t], lo2, hi2)]) for t in range(len(co_m))]
    imageio.mimsave(f"{args.out}/roi_pmip.gif", frames, duration=0.3, loop=0)
    try:
        import imageio_ffmpeg  # noqa: F401
        w = imageio.get_writer(f"{args.out}/roi_pmip.mp4", fps=8, codec="libx264", macro_block_size=1)
        for f in frames:
            w.append_data(np.stack([f]*3, -1))
        w.close()
    except Exception as e:
        print("mp4 skipped (conda install imageio-ffmpeg):", e)

    # trajectory QC
    ks = list(range(args.anchor, end))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks, inb); ax[0].set(title="fraction of ROI inside FOV (1=good)", xlabel="frame", ylabel="frac"); ax[0].set_ylim(0, 1.02)
    ax[1].plot(ks, disp); ax[1].set(title="ROI center displacement from anchor", xlabel="frame", ylabel="micron")
    plt.tight_layout(); plt.savefig(f"{args.out}/roi_trajectory.png", dpi=110)
    with open(f"{args.out}/roi_trajectory.csv", "w", newline="") as fh:
        wcsv = csv.writer(fh); wcsv.writerow(["frame", "frac_in_fov", "center_disp_um"])
        wcsv.writerows(zip(ks, inb, disp))
    print(f"\nWROTE {args.out}/roi_comoving.tif, roi_rawbox.tif, roi_pmip.gif(+mp4), roi_trajectory.png/.csv "
          f"[box {nx}x{ny}x{nz} vox @ {sp_o[0]:.2f}um]")
    print(f"min fraction-in-FOV = {min(inb):.2f} (if <~0.9 the region leaves the volume -> pick a more central center/shorter span)")

if __name__ == "__main__":
    main()
