#!/usr/bin/env python
"""
Follow a material ROI across the movie using the precomputed displacement fields, so the
selected region stays FIXED while the rest of the embryo moves. Reuses the fields/ and the
zarr cache produced by compute_fields_movie.py.

  For each frame k: compose the anchor->k field (incrementally, evaluated only on the small
  ROI grid -> cheap), then resample frame k into the ROI grid. The ROI box is defined once in
  the anchor frame; the field carries it through the contraction.

Outputs:
  roi_comoving.tif  (T,Z,Y,X)  the ROI followed (should look ~still when scrolling T)
  roi_rawbox.tif    (T,Z,Y,X)  the SAME fixed image-box, NOT followed (material drifts out)
  roi_pmip.gif/.mp4            pseudo z-MIP per frame: [raw fixed box | co-moving]

Example (server):
  python track_roi.py --input "H:\\...\\t*_Channel 1.tif" --fields-dir D:\\embryo_out\\fields ^
      --cache-dir D:\\embryo_cache --out D:\\embryo_out\\roi1 --xy-ds 4 --anchor 0 --size-um 60
"""
import argparse, glob, os
import numpy as np
import tifffile
import SimpleITK as sitk
import imageio.v2 as imageio
from compute_fields_movie import sitkimg, spacing, read_vol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--fields-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--xy-ds", type=int, default=4, help="MUST match the field resolution")
    ap.add_argument("--anchor", type=int, default=0)
    ap.add_argument("--center", default=None, help='"x,y,z" in microns; default = auto (bright-tissue centroid)')
    ap.add_argument("--size-um", type=float, default=60.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(args.input))
    cache = os.path.join(args.cache_dir, f"volcache_ds{args.xy_ds}.zarr") if args.cache_dir else None
    sp = spacing(args.xy_ds)                       # (x,y,z) microns at field resolution

    # --- choose ROI center (voxel x,y,z) ---
    va = read_vol(cache, files, args.anchor, args.xy_ds)   # (Z,Y,X)
    if args.center:
        cum = np.array([float(s) for s in args.center.split(",")])      # x,y,z microns
        cvox = cum / np.array(sp)
    else:                                          # intensity-weighted centroid of bright tissue
        thr = np.percentile(va, 90); m = va >= thr
        idx = np.argwhere(m).astype(np.float64); w = va[m].astype(np.float64)
        cz, cy, cx = (idx * w[:, None]).sum(0) / w.sum()
        cvox = np.array([cx, cy, cz])
    cphys = cvox * np.array(sp)
    print(f"ROI center voxel(x,y,z)={cvox.round(1)}  micron={cphys.round(1)}  size={args.size_um}um")

    # --- ROI grid in the anchor physical frame ---
    nx, ny, nz = (int(round(args.size_um / sp[i])) for i in range(3))
    half = args.size_um / 2.0
    roi = sitk.Image(nx, ny, nz, sitk.sitkFloat32)
    roi.SetSpacing(sp); roi.SetOrigin((cphys[0]-half, cphys[1]-half, cphys[2]-half))
    tff = sitk.TransformToDisplacementFieldFilter(); tff.SetReferenceImage(roi)

    def fld(i):
        return sitk.DisplacementFieldTransform(
            sitk.Cast(sitk.ReadImage(os.path.join(args.fields_dir, f"field_{i:04d}.nrrd")),
                      sitk.sitkVectorFloat64))

    ident = sitk.Transform(3, sitk.sitkIdentity)
    co, rawbox = [], []
    M = None                                       # composed anchor->k field, baked on the ROI grid
    for k in range(args.anchor, len(files)):
        volk = sitkimg(read_vol(cache, files, k, args.xy_ds), args.xy_ds)
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
        print(f"  ROI frame {k:04d}")
    co = np.stack(co); rawbox = np.stack(rawbox)    # (T,Z,Y,X)

    def u8(a, lo, hi):
        return (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    lo, hi = np.percentile(co, [1, 99.5])
    tifffile.imwrite(f"{args.out}/roi_comoving.tif", u8(co, lo, hi), imagej=True, metadata={"axes": "TZYX"})
    tifffile.imwrite(f"{args.out}/roi_rawbox.tif", u8(rawbox, lo, hi), imagej=True, metadata={"axes": "TZYX"})

    # pseudo z-MIP movie: [raw fixed box | co-moving]
    co_m, raw_m = co.max(1), rawbox.max(1)          # (T,Y,X)
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
    print(f"\nWROTE {args.out}/roi_comoving.tif, roi_rawbox.tif, roi_pmip.gif(+mp4)  "
          f"[box {nx}x{ny}x{nz} vox]  (left=raw fixed box, right=co-moving)")

if __name__ == "__main__":
    main()
