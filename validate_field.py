#!/usr/bin/env python
"""
OBJECTIVE field-accuracy metric (no eyeballing): nuclei-correspondence residual.

For a consecutive pair (frame A=i, B=i+1) with composed field_i (A->B):
  1. detect nuclei (local maxima of a smoothed volume) in A and in B,
  2. predict each A-nucleus position into B via the field,
  3. measure distance to the NEAREST detected nucleus in B  (TRE-like residual, in microns),
  4. compare to the IDENTITY baseline (no field) = how far nuclei actually moved.

A trustworthy field => median residual << identity, and on the order of the nucleus radius.
Reports microns, with the median nearest-neighbour nucleus spacing as the scale reference.

Run (server):
  python validate_field.py --input "H:\\...\\t*_Channel 1.tif" --fields-dir D:\\out\\fields ^
     --cache-dir D:\\cache --xy-ds 4 --pairs 0,13,14,16,30
"""
import argparse, glob, os
import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from compute_fields_movie import read_vol, spacing

def detect(vol, sp, sigma_um=3.0, pct=99.3, min_dist_um=6.0):
    """Return nuclei centroids as physical (x,y,z) microns. vol is (Z,Y,X); sp=(x,y,z)um."""
    sm = gaussian_filter(vol, [sigma_um/sp[2], sigma_um/sp[1], sigma_um/sp[0]])
    pk = peak_local_max(sm, min_distance=max(1, int(round(min_dist_um/sp[0]))),
                        threshold_abs=np.percentile(sm, pct))          # (n,3) z,y,x
    return np.stack([pk[:, 2]*sp[0], pk[:, 1]*sp[1], pk[:, 0]*sp[2]], 1)  # x,y,z um

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--fields-dir", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--xy-ds", type=int, default=4)
    ap.add_argument("--pairs", default=None, help="comma list of field indices i (A=i,B=i+1); default = a spread")
    args = ap.parse_args()
    files = sorted(glob.glob(args.input))
    cache = os.path.join(args.cache_dir, f"volcache_ds{args.xy_ds}.zarr") if args.cache_dir else None
    sp = spacing(args.xy_ds)
    pairs = ([int(x) for x in args.pairs.split(",")] if args.pairs
             else list(np.linspace(0, len(files)-2, 6).astype(int)))

    print(f"{'pair (A->B)':>14}{'nucA':>7}{'nucB':>7}{'NNspace':>9}{'identTRE':>10}{'fieldTRE':>10}{'%matched<1NN':>13}")
    for i in pairs:
        A = read_vol(cache, files, i, args.xy_ds)
        B = read_vol(cache, files, i+1, args.xy_ds)
        pa, pb = detect(A, sp), detect(B, sp)
        if len(pa) < 5 or len(pb) < 5:
            print(f"{i}->{i+1}: too few nuclei ({len(pa)},{len(pb)})"); continue
        tf = sitk.DisplacementFieldTransform(
            sitk.Cast(sitk.ReadImage(os.path.join(args.fields_dir, f"field_{i:04d}.nrrd")), sitk.sitkVectorFloat64))
        pred = np.array([tf.TransformPoint(tuple(map(float, p))) for p in pa])
        tree = cKDTree(pb)
        d_id = tree.query(pa)[0]; d_fld = tree.query(pred)[0]
        nn = np.median(tree.query(pb, k=2)[0][:, 1])      # median nearest-neighbour spacing in B
        frac = float(np.mean(d_fld < nn))                 # fraction landing within one NN-spacing of a real nucleus
        print(f"{f'{i}->{i+1}':>14}{len(pa):>7}{len(pb):>7}{nn:>9.1f}{np.median(d_id):>10.1f}{np.median(d_fld):>10.1f}{100*frac:>12.0f}%")
    print("\nunits = microns. fieldTRE should be << identTRE and ~<= NNspace (nucleus spacing).")

if __name__ == "__main__":
    main()
