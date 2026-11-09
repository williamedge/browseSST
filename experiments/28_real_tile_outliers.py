"""
Experiment 28: where are the heavy-tailed leave-out residuals of the real tile (exp 27)?  K-fold leave-out of the 28 x 28 px x 4 h Himawari tile with the exp-27 hyperparameters
fixed (every observed pixel is held out once, 5% at a time at random), z = (T - mean) / sqrt(var + noise).  Reports the tail fractions, whether the outliers are isolated pixels or
clusters, and how |z| relates to the local gradient and to the hour; saves the z cube and a figure.
Run: python experiments/28_real_tile_outliers.py [stride|mean] [folds]
"""
import importlib.util, json, pathlib, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec_ = importlib.util.spec_from_file_location("e27", ROOT / "experiments" / "27_real_tile_stage1.py")
e27 = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(e27)
H, TL = e27.H, e27.TL


def main(reduce='stride', folds=20):
    r = json.loads((ROOT / "data" / f"exp27_real_tile_{reduce}.json").read_text())
    cube = H.load_cube(e27.REAL, t0="2025-10-01T00:00:00", t1="2025-10-01T03:00:00", stride=2, reduce=reduce)
    f, j, i = e27.best_window(cube)
    spec = TL.TileSpec('real', 0, 4, j, j + 28, i, i + 28, 0, 4, j, j + 28, i, i + 28)
    td = TL.extract_tile(cube, spec)
    x, y, t, T = td.observed()
    n = x.size
    rng = np.random.default_rng(1)
    fold = rng.permutation(n) % folds
    z = np.full(n, np.nan); err = np.full(n, np.nan); sd = np.full(n, np.nan)
    t0 = time.time()
    for k in range(folds):
        hold = fold == k
        z[hold], err[hold], sd[hold] = e27.zscores(td, r, r['noise'], r['lr:mean'], hold)
    print(f"{folds}-fold leave-out, {n} pixels, {time.time() - t0:.0f}s ({reduce}, noise sd {np.sqrt(r['noise']):.4f} K)")
    absz = np.abs(z)
    print(f"  sd(z) {z.std():.2f}, |z| quantiles 50/90/99/99.9%: " + " / ".join(f"{q:.2f}" for q in np.quantile(absz, [0.5, 0.9, 0.99, 0.999])) + "  (Gaussian 0.67 / 1.64 / 2.58 / 3.29)")
    for c in (3, 4, 5):
        print(f"  fraction |z| > {c}: {np.mean(absz > c):.4f}  (Gaussian {2 * (1 - 0.5 * (1 + __import__('math').erf(c / 2 ** 0.5))):.5f})")
    sh = td.shape
    Z = td.scatter(z).reshape(sh); E = td.scatter(err).reshape(sh)
    big = np.abs(Z) > 3
    # isolated or clustered: neighbours (8-neighbourhood, same hour) that are also |z| > 3
    nb = np.zeros(sh, int)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            if dj or di:
                nb += np.roll(np.roll(big, dj, 1), di, 2)
    print(f"  |z|>3 points: {big.sum()}, of which with another |z|>3 neighbour in the same hour: {(big & (nb > 0)).sum()} (isolated: {(big & (nb == 0)).sum()})")
    print("  per hour: |z|>3 count " + ", ".join(str(int(big[k].sum())) for k in range(sh[0])) + f";  sd(z) per hour " + ", ".join(f"{np.nanstd(Z[k]):.2f}" for k in range(sh[0])))
    # relation with the local gradient of the observed field (centred differences on the anomaly)
    Tfull = np.where(td.obs, td.T, np.nan).reshape(sh)
    Gx = np.full(sh, np.nan); Gy = np.full(sh, np.nan)
    Gx[:, :, 1:-1] = (Tfull[:, :, 2:] - Tfull[:, :, :-2]) / 2; Gy[:, 1:-1, :] = (Tfull[:, 2:, :] - Tfull[:, :-2, :]) / 2
    Gm = np.hypot(Gx, Gy)
    ok = np.isfinite(Gm) & np.isfinite(Z)
    q = np.quantile(Gm[ok], [0.5, 0.9, 0.99])
    for lo, hi, nm in ((0, q[0], "gradient below median"), (q[0], q[1], "median-90%"), (q[1], q[2], "90-99%"), (q[2], 9, "top 1%")):
        m = ok & (Gm >= lo) & (Gm < hi)
        print(f"  {nm:24s}: n {m.sum():5d}, sd(z) {np.nanstd(Z[m]):.2f}, fraction |z|>3 {np.mean(np.abs(Z[m]) > 3):.4f}, rmse {np.sqrt(np.nanmean(E[m] ** 2)):.4f} K")
    np.savez(ROOT / "data" / f"exp28_real_tile_{reduce}_z.npz", z=Z, shape=np.array(sh))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 4, figsize=(15, 6.5), constrained_layout=True)
        for k in range(sh[0]):
            a = ax[0, k].pcolormesh(td.lon, td.lat, Tfull[k], cmap="turbo", shading="nearest"); ax[0, k].set_title(f"T anomaly (K), hour {k}")
            b = ax[1, k].pcolormesh(td.lon, td.lat, Z[k], cmap="RdBu_r", vmin=-5, vmax=5, shading="nearest"); ax[1, k].set_title(f"leave-out z, hour {k}")
        fig.colorbar(a, ax=ax[0], shrink=0.8); fig.colorbar(b, ax=ax[1], shrink=0.8)
        fig.savefig(ROOT / "data" / f"exp28_real_tile_{reduce}_z.png", dpi=90)
        print("figure saved")
    except Exception as e:                                       # plotting is optional
        print("no figure:", e)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else 'stride', int(a[1]) if len(a) > 1 else 20)
