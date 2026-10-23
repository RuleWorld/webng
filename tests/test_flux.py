"""
Tests for flux.py, which treats every line of tstates.txt as one sink.

Runs the real weFlux.run() with no WESTPA tools: each test writes a fluxanl.h5
in w_fluxanl's layout, stamped as up to date, so the analysis cache reuses it.
"""
import h5py
import numpy as np
import pytest

from webng.analysis.analysis import STAMP_ATTR
from webng.analysis.flux import weFlux

N_ITER = 40

# the fields of WESTPA's iter_block_ci_dtype, as w_fluxanl writes them
EVOL_DTYPE = np.dtype([("iter_start", np.uint32), ("iter_stop", np.uint32),
                       ("expected", np.float64), ("ci_lbound", np.float64),
                       ("ci_ubound", np.float64), ("sterr", np.float64),
                       ("corr_len", np.uint32)])


def _flux_analysis(tmp_path, monkeypatch, per_target, separate_cis):
    """
    weFlux over a fluxanl.h5 with one target per entry of `per_target`
    (per-iteration flux series). `separate_cis` are the per-target
    (mean, lo, hi) that w_fluxanl would have reported for each.
    """
    sim = tmp_path / "sim"
    (sim / "analysis").mkdir(parents=True)
    with h5py.File(sim / "west.h5", "w") as f:
        f.attrs["west_current_iteration"] = N_ITER + 1
        f.create_dataset("iterations/iter_00000001/pcoord", data=np.zeros((2, 3, 1)))
    monkeypatch.chdir(tmp_path)
    fx = weFlux({"sim_name": "sim", "work-path": None, "pcoords": ["A"], "step-iter": 5,
                 "output": "flux.png", "first-iter": None, "last-iter": None})

    n_iter = np.arange(1, N_ITER + 1)
    with h5py.File("fluxanl.h5", "w") as f:      # cwd is the analysis folder now
        avg = np.zeros(len(per_target), dtype=EVOL_DTYPE)
        for i, (fluxes, (mean, lo, hi)) in enumerate(zip(per_target, separate_cis)):
            g = f.create_group("target_flux/target_{}".format(i))
            g["n_iter"], g["flux"] = n_iter, np.asarray(fluxes)
            evol = np.zeros(3, dtype=EVOL_DTYPE)
            evol["iter_start"], evol["iter_stop"] = 1, [10, 20, 40]
            evol["expected"], evol["ci_lbound"], evol["ci_ubound"] = mean, lo, hi
            g["flux_evolution"] = evol
            avg[i] = (1, N_ITER + 1, mean, lo, hi, 0.0, 0)
        f["avg_flux"] = avg
        f.attrs[STAMP_ATTR] = fx.stamp("w_fluxanl", first_iter=int(fx.first_iter),
                                       last_iter=int(fx.last_iter), step_iter=fx.step_iter)
    return fx


def test_one_target_uses_w_fluxanls_own_results(tmp_path, monkeypatch):
    fluxes = np.random.default_rng(1).uniform(0, 1e-3, N_ITER)
    fx = _flux_analysis(tmp_path, monkeypatch, [fluxes], [(5e-4, 4e-4, 6e-4)])
    fig, ax = fx.run()
    assert fx.avg_flux == pytest.approx((5e-4, 4e-4, 6e-4))
    line = ax.lines[0]
    assert list(line.get_xdata()) == [10, 20, 40]
    assert list(line.get_ydata()) == pytest.approx([5e-4] * 3)


def test_several_target_lines_are_one_sink(tmp_path, monkeypatch):
    """
    Two bins of one sink whose fluxes swing in opposite directions: the total
    is nearly constant, so its CI must be nearly zero width -- while adding
    the two separate CIs' bounds would give a wide interval.
    """
    rng = np.random.default_rng(2)
    swing = rng.uniform(-0.5, 0.5, N_ITER)
    a = 1.0 + swing + rng.normal(0, 1e-3, N_ITER)
    b = 1.0 - swing + rng.normal(0, 1e-3, N_ITER)
    fx = _flux_analysis(tmp_path, monkeypatch, [a, b],
                        [(a.mean(), a.mean() - 0.2, a.mean() + 0.2),
                         (b.mean(), b.mean() - 0.2, b.mean() + 0.2)])
    fig, ax = fx.run()

    total = a + b
    mean, lo, hi = fx.avg_flux
    assert mean == pytest.approx(a.mean() + b.mean())        # means do add
    assert lo <= mean <= hi
    assert hi - lo < 0.02                                      # vs 0.8 from adding the bounds

    # evolution: cumulative mean of the TOTAL, in blocks of step-iter (5), as w_fluxanl does
    # (w_fluxanl counts iterations as last - first = 39 here, so 8 blocks, each
    # ending at min(first + 5*(k+1), last), each averaging the first 5*(k+1) iterations)
    line = ax.lines[0]
    assert list(line.get_xdata()) == [min(1 + 5 * (k + 1), N_ITER) for k in range(8)]
    assert list(line.get_ydata()) == pytest.approx(
        [total[:min(5 * (k + 1), N_ITER)].mean() for k in range(8)])
