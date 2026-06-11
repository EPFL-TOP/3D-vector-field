"""
Characterize Defective_somite.tif WITHOUT loading the full 7GB.
Reads individual TIFF pages (ImageJ hyperstack XYCZT order).

Outputs numeric stats + PNGs into investigation/out/ and a JSON summary
(best channel, best-focus z) for reuse by the CoTracker3 empirical step.
"""
import json
import os

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cv2
    HAVE_CV2 = True
except Exception as e:  # pragma: no cover
    HAVE_CV2 = False
    print("cv2 unavailable:", e)

F = "Defective_somite.tif"
OUT = "investigation/out"
os.makedirs(OUT, exist_ok=True)

T, Z, C, Y, X = 179, 71, 3, 303, 303  # axes TZCYX


def page_index(t, z, c):
    # ImageJ hyperstack page order is XYCZT -> C fastest, then Z, then T
    return (t * Z + z) * C + c


# >4GB ImageJ hyperstacks store one IFD + contiguous data -> use a lazy memmap
ARR = tifffile.memmap(F)  # (T,Z,C,Y,X) uint16, not loaded into RAM
assert ARR.shape == (T, Z, C, Y, X), ARR.shape


def get(t, z, c):
    return np.asarray(ARR[t, z, c], dtype=np.float32)


def norm8(img, lo=1, hi=99):
    a, b = np.percentile(img, [lo, hi])
    if b <= a:
        b = a + 1
    return np.clip((img - a) / (b - a), 0, 1)


def lapvar(img):
    if HAVE_CV2:
        return float(cv2.Laplacian(img, cv2.CV_32F).var())
    gy, gx = np.gradient(img)
    return float(np.var(gx) + np.var(gy))


def grad_energy(img):
    gy, gx = np.gradient(img)
    return float(np.mean(gx ** 2 + gy ** 2))


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    da, db = a.std(), b.std()
    if da < 1e-6 or db < 1e-6:
        return 0.0
    return float(np.mean((a / da) * (b / db)))


print("\n=== per-channel stats (sampled t,z) ===")
print(f"{'t':>4}{'z':>4}{'c':>3}{'mean':>10}{'std':>10}{'p99':>8}{'lapvar':>12}{'gradE':>12}")
chan_score = np.zeros(C)
for t in [0, 90, 178]:
    for z in [20, 35, 50]:
        for c in range(C):
            im = get(t, z, c)
            lv, ge = lapvar(im), grad_energy(im)
            chan_score[c] += lv
            print(f"{t:>4}{z:>4}{c:>3}{im.mean():>10.1f}{im.std():>10.1f}"
                  f"{np.percentile(im,99):>8.0f}{lv:>12.1f}{ge:>12.1f}")
best_c = int(np.argmax(chan_score))
print(f"\n=> best-texture channel (sum lapvar): c={best_c}  scores={chan_score.tolist()}")

# ---- focus-vs-z curves per channel at t=0 ----
print("\n=== focus (lapvar) vs z at t=0 ===")
focus = np.zeros((C, Z))
for c in range(C):
    for z in range(Z):
        focus[c, z] = lapvar(get(0, z, c))
best_focus_z = int(np.argmax(focus[best_c]))
plt.figure(figsize=(7, 4))
for c in range(C):
    plt.plot(range(Z), focus[c] / focus[c].max(), label=f"ch{c}")
plt.axvline(best_focus_z, color="k", ls="--", lw=0.8, label=f"best z (ch{best_c})={best_focus_z}")
plt.xlabel("z slice"); plt.ylabel("normalized lapvar (focus)"); plt.legend()
plt.title("Focus vs z (t=0)"); plt.tight_layout()
plt.savefig(f"{OUT}/focus_vs_z.png", dpi=110); plt.close()
print(f"best-focus z for ch{best_c}: {best_focus_z}")

# ---- channel montage at t=0, best focus z ----
fig, ax = plt.subplots(1, C, figsize=(3 * C, 3))
for c in range(C):
    ax[c].imshow(norm8(get(0, best_focus_z, c)), cmap="gray")
    ax[c].set_title(f"ch{c} t0 z{best_focus_z}"); ax[c].axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/channels_t0.png", dpi=110); plt.close()

# ---- time montage at best channel + best focus z ----
ts = np.linspace(0, T - 1, 8).astype(int)
fig, ax = plt.subplots(1, len(ts), figsize=(2.2 * len(ts), 2.4))
for i, t in enumerate(ts):
    ax[i].imshow(norm8(get(t, best_focus_z, best_c)), cmap="gray")
    ax[i].set_title(f"t{t}"); ax[i].axis("off")
plt.suptitle(f"ch{best_c} z{best_focus_z} over time"); plt.tight_layout()
plt.savefig(f"{OUT}/time_montage.png", dpi=110); plt.close()

# ---- in-plane motion magnitude (phase correlation, consecutive t) ----
print("\n=== in-plane drift (phaseCorrelate consecutive t, ch%d z%d) ===" % (best_c, best_focus_z))
shifts = []
if HAVE_CV2:
    prev = norm8(get(0, best_focus_z, best_c)).astype(np.float32)
    win = cv2.createHanningWindow((X, Y), cv2.CV_32F)
    for t in range(1, T):
        cur = norm8(get(t, best_focus_z, best_c)).astype(np.float32)
        (dx, dy), resp = cv2.phaseCorrelate(prev, cur, win)
        shifts.append((dx, dy, resp))
        prev = cur
    sh = np.array(shifts)
    cum = np.cumsum(sh[:, :2], axis=0)
    print(f"per-step |shift| px: mean={np.hypot(sh[:,0],sh[:,1]).mean():.2f} "
          f"max={np.hypot(sh[:,0],sh[:,1]).max():.2f}")
    print(f"cumulative drift over T: dx={cum[-1,0]:.1f} dy={cum[-1,1]:.1f} px "
          f"(range x[{cum[:,0].min():.1f},{cum[:,0].max():.1f}] "
          f"y[{cum[:,1].min():.1f},{cum[:,1].max():.1f}])")
    plt.figure(figsize=(6, 4))
    plt.plot(cum[:, 0], label="cum dx"); plt.plot(cum[:, 1], label="cum dy")
    plt.xlabel("t"); plt.ylabel("cumulative shift (px)"); plt.legend()
    plt.title("In-plane cumulative drift"); plt.tight_layout()
    plt.savefig(f"{OUT}/inplane_drift.png", dpi=110); plt.close()

# ---- z-matching feasibility: NCC(z0 at t, z1 at t+dt) ----
# This is the CRUX of the user's idea: can image similarity match z across time?
def zmatch(dt, tag):
    s0 = np.stack([norm8(get(0, z, best_c)) for z in range(Z)])
    s1 = np.stack([norm8(get(dt, z, best_c)) for z in range(Z)])
    M = np.zeros((Z, Z))
    for i in range(Z):
        for j in range(Z):
            M[i, j] = ncc(s0[i], s1[j])
    best_j = M.argmax(axis=1)
    offs = best_j - np.arange(Z)
    print(f"\n=== z-match t0 vs t{dt} (ch{best_c}) ===")
    print(f"median best-match z-offset: {np.median(offs):.1f}  "
          f"(per-z offset range [{offs.min()},{offs.max()}])")
    print(f"mean max-NCC per z0: {M.max(axis=1).mean():.3f}  "
          f"mean diagonal NCC: {np.mean(np.diag(M)):.3f}")
    plt.figure(figsize=(5, 4.2))
    plt.imshow(M, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(label="NCC"); plt.xlabel(f"z at t{dt}"); plt.ylabel("z at t0")
    plt.plot(best_j, range(Z), "r.", ms=3, label="argmax")
    plt.plot(range(Z), range(Z), "w--", lw=0.6, label="diagonal")
    plt.legend(loc="lower right", fontsize=7)
    plt.title(f"z-similarity NCC  t0 vs t{dt}"); plt.tight_layout()
    plt.savefig(f"{OUT}/zmatch_t{dt}.png", dpi=110); plt.close()
    return float(np.median(offs)), float(M.max(axis=1).mean())

off1, ncc1 = zmatch(1, "t1")
off5, ncc5 = zmatch(5, "t5")

summary = dict(
    shape=[T, Z, C, Y, X], best_channel=best_c, best_focus_z=best_focus_z,
    channel_lapvar_scores=chan_score.tolist(),
    inplane_step_shift_mean=float(np.hypot(sh[:,0],sh[:,1]).mean()) if HAVE_CV2 else None,
    inplane_step_shift_max=float(np.hypot(sh[:,0],sh[:,1]).max()) if HAVE_CV2 else None,
    inplane_cum_drift=[float(cum[-1,0]), float(cum[-1,1])] if HAVE_CV2 else None,
    zmatch_median_offset_t1=off1, zmatch_meanmaxNCC_t1=ncc1,
    zmatch_median_offset_t5=off5, zmatch_meanmaxNCC_t5=ncc5,
)
with open(f"{OUT}/data_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
print(f"\nPNGs + summary written to {OUT}/")
