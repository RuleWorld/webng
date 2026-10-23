"""
Unit tests for the MABL driver that `webng setup` generates (MABL_driver.py).

The generated driver runs against stand-ins for the three WESTPA methods it
calls (_split_walker, _merge_walkers, and the pre/post checks), so these test
MABL's own decisions -- which walkers it splits and merges, and that walker
count and total weight are conserved -- without running a simulation.
"""
import importlib.util
import math

import numpy as np
import pytest


class Walker:
    """Stand-in for a WESTPA Segment: just what MABLDriver reads."""
    _next_id = 0

    def __init__(self, weight, pcoord, parent_id=0, n_iter=2):
        self.weight = weight
        self.pcoord = np.array(pcoord, dtype=float)    # (frames, dims)
        self.parent_id = parent_id
        self.n_iter = n_iter
        self.seg_id = Walker._next_id
        Walker._next_id += 1


def _load_driver(bare_weconvert, tmp_path, monkeypatch, **settings):
    opts = dict(start=[0.0, 0.0], target=[10.0, 10.0], coord_weights=[1, 1],
                n_split=2, weight_cap_factor=10.0, recycle_region=None)
    opts.update(settings)
    monkeypatch.chdir(tmp_path)
    bare_weconvert(**opts)._write_mabl_driver()
    spec = importlib.util.spec_from_file_location("MABL_driver", tmp_path / "MABL_driver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class Driver(mod.MABLDriver):
        """The generated driver, with WESTPA's split/merge replaced by stand-ins
        that conserve weight exactly the way WESTPA's do."""
        def _split_walker(self, segment, m, bin):
            self.split_log.append(segment)
            return [Walker(segment.weight / m, segment.pcoord.copy(), segment.parent_id)
                    for _ in range(m)]

        def _merge_walkers(self, segments, cumul_weight, bin):
            self.merge_log.append(list(segments))
            survivor = max(segments, key=lambda s: s.weight)
            merged = Walker(sum(s.weight for s in segments), survivor.pcoord.copy(),
                            survivor.parent_id)
            return merged, survivor

        def _check_pre(self):
            pass

        def _check_post(self):
            pass

    return mod, Driver


def _driver(Driver, walkers, moved=True):
    """A driver about to resample `walkers` (all in bin 0)."""
    d = Driver.__new__(Driver)          # skip WEDriver.__init__ (needs a WESTPA rc)
    d.split_log, d.merge_log = [], []
    d.target_states, d.avail_initial_states, d.used_initial_states = {}, {}, {}
    d.new_weights = None
    # last iteration's walkers: their first and last frames differ if they moved
    current = []
    for w in walkers:
        pc = w.pcoord.copy()
        if moved:
            pc[-1] = pc[-1] + 1.0
        current.append(Walker(w.weight, pc))
    d.final_binning = [set(current)]
    d.next_iter_binning = [set(walkers), set()]
    return d


def _ensemble(n, rng=None):
    rng = rng or np.random.default_rng(0)
    pts = rng.uniform(0, 10, size=(n, 2))
    return [Walker(1.0 / n, [p, p]) for p in pts]


def _totals(d):
    walkers = [w for b in d.next_iter_binning for w in b]
    return len(walkers), sum(w.weight for w in walkers)


def test_walker_count_and_weight_conserved(bare_weconvert, tmp_path, monkeypatch):
    _, Driver = _load_driver(bare_weconvert, tmp_path, monkeypatch, n_split=5)
    d = _driver(Driver, _ensemble(40))
    d._run_we()
    n, w = _totals(d)
    assert n == 40
    assert w == pytest.approx(1.0)
    assert len(d.split_log) == 5 and len(d.merge_log) == 5


def test_splits_the_highest_scoring_walkers(bare_weconvert, tmp_path, monkeypatch):
    mod, Driver = _load_driver(bare_weconvert, tmp_path, monkeypatch, n_split=2)
    walkers = _ensemble(20)
    d = _driver(Driver, walkers)
    segments, pcoords = d._rebuild(d.next_iter_binning[0])
    scores = d._compute_split_scores(segments, d._compute_progress(pcoords, 2))
    expected = {segments[i] for i in np.argsort(-scores)[:2]}
    d._run_we()
    assert set(d.split_log) == expected


def test_coord_weights_change_the_ranking(bare_weconvert, tmp_path, monkeypatch):
    """Weights must matter (they didn't when progress was a product)."""
    a = Walker(0.5, [[9.0, 1.0], [9.0, 1.0]])    # far along dimension 0
    b = Walker(0.5, [[1.0, 9.0], [1.0, 9.0]])    # far along dimension 1
    for weights, winner in [([5, 1], a), ([1, 5], b)]:
        sub = tmp_path / "w{}{}".format(*weights)
        sub.mkdir()
        _, Driver = _load_driver(bare_weconvert, sub, monkeypatch, coord_weights=weights)
        d = _driver(Driver, [a, b])
        segments, pcoords = d._rebuild(d.next_iter_binning[0])
        progress = d._compute_progress(pcoords, 2)
        assert segments[int(np.argmax(progress))] is winner


def test_initialization_check_looks_at_every_pcoord(bare_weconvert, tmp_path, monkeypatch):
    _, Driver = _load_driver(bare_weconvert, tmp_path, monkeypatch)
    d = _driver(Driver, _ensemble(4), moved=False)
    assert d._is_initialization()
    # only the SECOND coordinate moved: that's still a real iteration
    for w in d.final_binning[0]:
        w.pcoord[-1, 1] += 1.0
    assert not d._is_initialization()


def test_odd_merge_count_never_merges_a_walker_twice(bare_weconvert, tmp_path, monkeypatch):
    """Too few mergeable walkers gives an odd selection; weight must still be conserved."""
    _, Driver = _load_driver(bare_weconvert, tmp_path, monkeypatch, n_split=4)
    walkers = _ensemble(9)
    for w in walkers[:4]:
        w.parent_id = -1             # just recycled: never merged
    d = _driver(Driver, walkers)
    d._run_we()
    merged = [w for group in d.merge_log for w in group]
    assert len(merged) == len(set(merged))
    assert _totals(d)[1] == pytest.approx(1.0)


def test_weight_cap_splits_heavy_walkers(bare_weconvert, tmp_path, monkeypatch):
    _, Driver = _load_driver(bare_weconvert, tmp_path, monkeypatch, n_split=0, weight_cap_factor=2.0)
    walkers = [Walker(0.01, [[1, 1], [1, 1]]) for _ in range(9)]
    heavy = Walker(0.91, [[2, 2], [2, 2]])
    d = _driver(Driver, walkers + [heavy])
    d._run_we()
    cap = 2.0 / 10                              # factor x average weight
    assert d.split_log[0] is heavy
    assert len(d.merge_log) == math.ceil(0.91 / cap) - 1    # merges undo the extras
    assert _totals(d) == (10, pytest.approx(1.0))


def test_recycle_region_bounds(bare_weconvert, tmp_path, monkeypatch):
    _, Driver = _load_driver(bare_weconvert, tmp_path, monkeypatch,
                             recycle_region=[[8.0, None], [None, 2.0]])
    d = _driver(Driver, _ensemble(2))
    assert d._in_recycle_region([9.0, 1.0])
    assert not d._in_recycle_region([7.0, 1.0])     # below the first minimum
    assert not d._in_recycle_region([9.0, 3.0])     # above the second maximum
    off = tmp_path / "off"
    off.mkdir()
    _, NoRecycle = _load_driver(bare_weconvert, off, monkeypatch, recycle_region=None)
    assert not _driver(NoRecycle, _ensemble(2))._in_recycle_region([9.0, 1.0])
