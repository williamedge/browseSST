"""
Experiment 26: independent overlapping tiles blended with a partition of unity, versus a single job (seam test).

Domain: nx x nx px (default 36, i.e. 144 km) x nt h, the fronts + filament field (the fronts sit in the south-west part, so the tiles differ in character), sigma_T = 0.03 K.
Hyperparameters are SHARED and held fixed, as planned for production (estimated once, applied everywhere): stage 1 = the exp-21 fronts fit at 0.03 K (5/2 : 5/2) from another window of the
same statistics; stage 2 = the true Helmholtz hyperparameters with a fixed noise variance (2.17, the exp-22 equation-error value). So the test isolates tiling / blending, not estimation.
Each job: latent T and Tt posterior means from the tile's own data (including its buffer), 4th-order gradients, stage-2 velocity from the products, predicted at EVERY grid point of the
tile (mean, variance). 'ref' = one job over the whole domain; 'bK' = 2 x 2 tiles with cores nx/2 and a buffer of K px on the sides that border another tile.
Blending: weight 1 on the core, sin^2 taper over the buffer (0 at the outer edge), normalised by the sum of weights; K = 0 is the hard partition.
Run: python experiments/26_tiling.py job <ref|b0|b2|b4> [nx] [nt]      |      python experiments/26_tiling.py blend [nx] [nt]
"""
import importlib.util, json, pathlib, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / (name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


e21 = _load("21_fronts_filaments"); e13, G, K, ZS, S = e21.e13, e21.G, e21.K, e21.ZS, e21.S
from inversion_sst_gp.spacetime.impute import grid_gradients
from inversion_sst_gp.spacetime.model import LowRankTerm, SpaceTimeGP, diurnal_harmonics

NOISE_S2 = 2.17


def make_config(nx, nt):
    kw = dict(e21.FRONT_CFG); kw.update(nx=nx, ny=nx, n_hours=nt)
    return S.SyntheticConfig(**kw)


def load_domain(nx, nt, sigma_T=0.03):
    cfg = make_config(nx, nt)
    ds0 = S.load_or_generate(cfg, ROOT / "data" / f"synth_{nx}px_{nt}h_fronts.nc")
    return cfg, ds0, S.reobserve(ds0, sigma_T, seed=cfg.seed + 7)


def run_tile(cfg, ds, x0, x1, y0, y1, nnum_pred=200):
    """Latent-T + stage-2 velocity on the sub-cube [y0:y1, x0:x1]; returns (u, v, var_u, var_v) on that sub-grid, plus timings."""
    tim = {}
    T = ds['T_obs'].values[:, y0:y1, x0:x1]
    nt, ny, nx = T.shape
    xs, ys = ds['x'].values[x0:x1], ds['y'].values[y0:y1]
    tt, yy, xx = np.meshgrid(ds['time'].values, ys, xs, indexing='ij')
    x, y, t = xx.ravel(), yy.ravel(), tt.ravel()
    r = json.loads((ROOT / "data" / "exp21_fit_sigma0.03_nu2.5_2.5.json").read_text())
    t0 = time.time()
    p0 = G.pack_params_T_res(r['sigma1'], r['l1'], r['tau1'], r['sigma2'], r['l2'], r['tau2'], 2.5, 2.5, 2.5)
    gp = G.TGP(G.build_T_points(x, y, t), T.ravel(), p0, r['noise'], lowrank=e13.T_terms(r['lr:mean'], r['lr:diurnal']), nnum=100, Lx=20e3, Lt=24 * 3600.0, covfunc=G.T_covfunc_res)
    nn = gp.pred_neighbours(x, y, t, nnum_pred)
    tim['stage1_build'] = time.time() - t0
    t0 = time.time()
    Tm = gp.predict('T', x, y, t, nn_array=nn)[0]
    Ttm = gp.predict('Tt', x, y, t, nn_array=nn)[0]
    tim['stage1_predict'] = time.time() - t0
    Tx, Ty = grid_gradients(Tm, (nt, ny, nx), float(cfg.dx), order=4)
    valid = np.isfinite(Tx) & np.isfinite(Ty)
    tp = cfg.true_params(ZS)
    p_vel = K.pack_params(tp['sigma_psi'], tp['l_psi'], tp['sigma_phi'], tp['l_phi'], tp['tau0_psi'], tp['tau0_phi'], tp['sigma_S'], tp['l_S'], tp['tau0_S'])
    t0 = time.time()
    pts = K.build_points(x[valid], y[valid], t[valid], Tx[valid] / ZS, Ty[valid] / ZS)
    gp2 = SpaceTimeGP(pts, Ttm[valid] / ZS, p_vel, NOISE_S2, lowrank=[LowRankTerm('diurnal', {'S': diurnal_harmonics(4)}, scale=0.5)], nnum=100)
    nn2 = gp2.pred_neighbours(x, y, t, nnum_pred)
    tim['stage2_build'] = time.time() - t0
    t0 = time.time()
    u, vu = gp2.predict('u', x, y, t, nn_array=nn2)
    v, vv = gp2.predict('v', x, y, t, nn_array=nn2)
    tim['stage2_predict'] = time.time() - t0
    tim['N'] = int(len(x)); tim['N_stage2'] = int(valid.sum())
    sh = (nt, ny, nx)
    return u.reshape(sh), v.reshape(sh), vu.reshape(sh), vv.reshape(sh), tim


def tile_layout(nx, b):
    """List of (name, buffered extents, core extents) for the 2 x 2 tiling with buffer b px (indices into the domain)."""
    h = nx // 2
    out = []
    for j, (cy0, cy1) in enumerate(((0, h), (h, nx))):
        for i, (cx0, cx1) in enumerate(((0, h), (h, nx))):
            out.append((f"t{j}{i}", (max(0, cx0 - b), min(nx, cx1 + b), max(0, cy0 - b), min(nx, cy1 + b)), (cx0, cx1, cy0, cy1)))
    return out


def job(name, nx, nt):
    cfg, ds0, ds = load_domain(nx, nt)
    print(f"[{name}] domain {nx}x{nx} px x {nt} h, N = {nx * nx * nt}", flush=True)
    if name == 'ref':
        tiles = [("ref", (0, nx, 0, nx), (0, nx, 0, nx))]
    else:
        tiles = tile_layout(nx, int(name[1:]))
    for tname, (x0, x1, y0, y1), core in tiles:
        t0 = time.time()
        u, v, vu, vv, tim = run_tile(cfg, ds, x0, x1, y0, y1)
        np.savez(ROOT / "data" / f"exp26_{name}_{tname}_{nx}px_{nt}h.npz", u=u, v=v, vu=vu, vv=vv, ext=np.array([x0, x1, y0, y1]), core=np.array(core), timing=json.dumps(tim))
        print(f"[{name}/{tname}] extent x {x0}:{x1} y {y0}:{y1} N {tim['N']}: " + ", ".join(f"{k} {v_:.0f}s" for k, v_ in tim.items() if k not in ('N', 'N_stage2')) + f", total {time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)


def taper_1d(n, lo_buf, hi_buf):
    """Weights along one axis for a tile of n cells with lo_buf / hi_buf buffer cells on the low / high side (0 = no taper there)."""
    w = np.ones(n)
    for k in range(lo_buf):
        w[k] = np.sin(0.5 * np.pi * (k + 0.5) / lo_buf) ** 2
    for k in range(hi_buf):
        w[n - 1 - k] = np.sin(0.5 * np.pi * (k + 0.5) / hi_buf) ** 2
    return w


def blend(nx, nt):
    cfg, ds0, ds = load_domain(nx, nt)
    truth_u, truth_v = ds0['u_true'].values, ds0['v_true'].values
    T_true = ds0['T_true'].values
    g = S.derivative_products(T_true, float(cfg.dx), order=4)
    gm = np.hypot(g[1], g[2])
    rim = 3                                           # score away from the domain boundary and the first / last hours
    score = np.zeros((nt, nx, nx), bool); score[2:nt - 2, rim:nx - rim, rim:nx - rim] = True
    h = nx // 2
    yy, xx = np.meshgrid(np.arange(nx), np.arange(nx), indexing='ij')
    band = (np.abs(xx - (h - 0.5)) <= 3) | (np.abs(yy - (h - 0.5)) <= 3)     # within 3 cells of a seam
    band = np.broadcast_to(band, (nt, nx, nx))
    results = {}
    files = {p.name for p in (ROOT / "data").glob(f"exp26_*_{nx}px_{nt}h.npz")}

    def load(name, tname):
        z = np.load(ROOT / "data" / f"exp26_{name}_{tname}_{nx}px_{nt}h.npz", allow_pickle=True)
        return z, json.loads(str(z['timing']))

    out = {}
    if f"exp26_ref_ref_{nx}px_{nt}h.npz" in files:
        z, tim = load('ref', 'ref'); out['single job'] = (z['u'], z['v'], z['vu'], z['vv'], tim['N'])
    for name in ('b0', 'b2', 'b4'):
        b = int(name[1:])
        if not all(f"exp26_{name}_{t}_{nx}px_{nt}h.npz" in files for t in ('t00', 't01', 't10', 't11')):
            continue
        num = [np.zeros((nt, nx, nx)) for _ in range(4)]; den = np.zeros((nt, nx, nx)); Ns = []
        for tname, (x0, x1, y0, y1), (cx0, cx1, cy0, cy1) in tile_layout(nx, b):
            z, tim = load(name, tname); Ns.append(tim['N'])
            wx = taper_1d(x1 - x0, cx0 - x0, x1 - cx1); wy = taper_1d(y1 - y0, cy0 - y0, y1 - cy1)
            w = (wy[:, None] * wx[None, :])[None]
            if b == 0:
                w = np.ones_like(w)                    # hard partition: the core only
            for k, key in enumerate(('u', 'v', 'vu', 'vv')):
                num[k][:, y0:y1, x0:x1] += w * z[key]
            den[:, y0:y1, x0:x1] += w
        out[f"tiles, buffer {b} px"] = (num[0] / den, num[1] / den, num[2] / den, num[3] / den, max(Ns))
    print(f"seam test, {nx}x{nx} px x {nt} h ({len(T_true.ravel())} points), scored on {score.sum()} points ({int((score & band).sum())} within 3 cells of a seam)\n")
    ref = out.get('single job')
    print(f"{'estimator':<22s} {'N/job':>7s} | u skill all / seam / away | v skill all / seam / away | rms diff to single job (seam)  / posterior sd of single job (seam)")
    for label, (u, v, vu, vv, N) in out.items():
        cells = []
        for est, tr in ((u, truth_u), (v, truth_v)):
            def sk(m): e = (est - tr)[m]; return 1 - np.mean(e ** 2) / tr[m].var()
            cells.append(f"{sk(score):+.3f} / {sk(score & band):+.3f} / {sk(score & ~band):+.3f}")
        extra = ""
        if ref is not None and label != 'single job':
            m = score & band
            du = np.sqrt(np.mean((u - ref[0])[m] ** 2)); dv = np.sqrt(np.mean((v - ref[1])[m] ** 2))
            extra = f" | u {du:.4f} v {dv:.4f} / u {np.sqrt(np.mean(ref[2][m])):.4f} v {np.sqrt(np.mean(ref[3][m])):.4f}"
        print(f"{label:<22s} {N:7d} | {cells[0]} | {cells[1]}{extra}")
    # blended coverage with the (blended) variance as a rough indication
    print("\n95% coverage (blended mean, blended variance): " + "; ".join(
        f"{label}: u {np.mean((np.abs(u - truth_u) < 1.96 * np.sqrt(vu))[score]):.2f} v {np.mean((np.abs(v - truth_v) < 1.96 * np.sqrt(vv))[score]):.2f}" for label, (u, v, vu, vv, N) in out.items()))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == 'job':
        job(a[1], int(a[2]) if len(a) > 2 else 36, int(a[3]) if len(a) > 3 else 48)
    else:
        blend(int(a[1]) if len(a) > 1 else 36, int(a[2]) if len(a) > 2 else 48)
