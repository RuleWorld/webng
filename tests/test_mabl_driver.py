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
                n_split=2, weight_cap_factor=10.0, recycle_region=None,
                log_scale=[False, False], merge_progress_floor=0.01,
                local_merge=False, merge_pool=4, merge_log=False)
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


# ---------------------------------------------------------------------------
# The driver is webng/assets/MABL_driver.py with its settings block filled in
# ---------------------------------------------------------------------------

def test_driver_is_the_asset_with_its_settings_filled(bare_weconvert, tmp_path, monkeypatch):
    from webng.core.weConvert import weConvert
    mod, _ = _load_driver(bare_weconvert, tmp_path, monkeypatch, start=[100, 3250], target=[3350, 100],
                          coord_weights=[1, 2], log_scale=[True, False], n_split=7,
                          merge_progress_floor=0.05, local_merge=True, merge_pool=3, merge_log=True,
                          recycle_region=[[1001, None], [None, 499]])
    assert (mod.MABL_PCOORDS, mod.MABL_START, mod.MABL_TARGET) == (["NANOG", "GATA6"], [100, 3250], [3350, 100])
    assert (mod.MABL_COORD_WEIGHTS, mod.MABL_LOG_SCALE, mod.MABL_N_SPLIT) == ([1, 2], [True, False], 7)
    assert (mod.MABL_MERGE_PROGRESS_FLOOR, mod.MABL_LOCAL_MERGE, mod.MABL_MERGE_POOL, mod.MABL_MERGE_LOG) == \
        (0.05, True, 3, True)
    assert mod.MABL_RECYCLE_REGION == [[1001, None], [None, 499]]
    # outside the settings block the file is the asset, byte for byte
    asset = open(weConvert._assets_dir() + "/MABL_driver.py").read()
    written = (tmp_path / "MABL_driver.py").read_text()
    b, e = weConvert.MABL_CONFIG_BEGIN, weConvert.MABL_CONFIG_END
    assert written.split(b)[0] == asset.split(b)[0]
    assert written.split(e)[1] == asset.split(e)[1]


def test_log_scale_progress(bare_weconvert, tmp_path, monkeypatch):
    """CDX2-like coordinate linear (falls 3250 -> 100), OCT4-like log (rises 100 -> 3350)."""
    _, Driver = _load_driver(bare_weconvert, tmp_path, monkeypatch, start=[3250, 100], target=[100, 3350],
                             log_scale=[False, True])
    d = _driver(Driver, _ensemble(2))
    pc = np.array([[[3250, 100]], [[2000, 300]], [[100, 3350]], [[3500, 50]], [[50, 5000]]], dtype=float)
    S = d._compute_progress(pc, 2)
    assert S == pytest.approx([0.0,
                               (1 - 1900 / 3150) + np.log(3) / np.log(33.5),
                               2.0,
                               0.0,          # beyond the start on both: clipped at 0
                               (1 - 50 / 3150) + 1.0])   # beyond the target: log clipped at 1


def test_merge_floor_ranks_zero_progress_walkers_by_weight(bare_weconvert, tmp_path, monkeypatch):
    heavy_zero = Walker(0.3, [[0, 0], [0, 0]])          # zero progress, heavy
    light = Walker(0.001, [[5, 5], [5, 5]])              # some progress, light
    for floor, first in [(0.01, light), (1e-300, heavy_zero)]:
        sub = tmp_path / "f{}".format(floor)
        sub.mkdir()
        _, Driver = _load_driver(bare_weconvert, sub, monkeypatch, merge_progress_floor=floor)
        d = _driver(Driver, [heavy_zero, light])
        segs = [heavy_zero, light]
        pcs = np.array([s.pcoord for s in segs])
        ghost = d._compute_ghost_scores(segs, d._compute_progress(pcs, 2))
        assert segs[int(np.argmax(ghost))] is first


def _two_clusters(n_each=4):
    """Candidates alternating between two far-apart clusters in merge-score rank."""
    segs, pcs = [], []
    for i in range(2 * n_each):
        x = 0.0 if i % 2 == 0 else 100.0
        segs.append(Walker(0.01, [[x + 0.1 * i, x], [x + 0.1 * i, x]]))
        pcs.append(segs[-1].pcoord)
    ghost = np.arange(2 * n_each, 0, -1, dtype=float)     # rank = index order
    return np.array(segs, dtype=object), np.array(pcs), ghost


def test_local_merge_pairs_neighbours_rank_merge_does_not(bare_weconvert, tmp_path, monkeypatch):
    _, Driver = _load_driver(bare_weconvert, tmp_path, monkeypatch)
    d = _driver(Driver, _ensemble(2))
    segs, pcs, ghost = _two_clusters()
    cluster = lambda i: int(pcs[i, 0, 1] > 50)
    rank = d._rank_merge_groups(segs, ghost, 4)
    local = d._local_merge_groups(segs, pcs, ghost, 4)
    assert all(cluster(a) != cluster(b) for a, b in rank)          # ranks alternate clusters
    assert all(cluster(a) == cluster(b) for a, b in local)
    assert len(local) == 4 and sorted(i for g in local for i in g) == list(range(8))


def test_local_merge_pool_limits_partners(bare_weconvert, tmp_path, monkeypatch):
    _, Driver = _load_driver(bare_weconvert, tmp_path, monkeypatch, merge_pool=2)
    d = _driver(Driver, _ensemble(2))
    segs, pcs, ghost = _two_clusters(n_each=6)
    groups = d._local_merge_groups(segs, pcs, ghost, 2)
    assert sorted(i for g in groups for i in g) == [0, 1, 2, 3]    # pool = top 2 x 2 by merge score


def test_local_merge_odd_leftover_merges_each_walker_once(bare_weconvert, tmp_path, monkeypatch):
    _, Driver = _load_driver(bare_weconvert, tmp_path, monkeypatch)
    d = _driver(Driver, _ensemble(2))
    segs, pcs, ghost = _two_clusters()
    segs = segs[:5]; pcs = pcs[:5]; ghost = ghost[:5]
    groups = d._local_merge_groups(segs, pcs, ghost, 3)            # wants 3 merges from 5 walkers
    members = [i for g in groups for i in g]
    assert len(members) == len(set(members)) == 5
    assert sum(len(g) - 1 for g in groups) == 3


def test_local_merge_conserves_walkers_and_weight(bare_weconvert, tmp_path, monkeypatch):
    _, Driver = _load_driver(bare_weconvert, tmp_path, monkeypatch, n_split=5, local_merge=True)
    d = _driver(Driver, _ensemble(40))
    d._run_we()
    assert _totals(d) == (40, pytest.approx(1.0))
    assert len(d.merge_log) == 5 and all(len(g) == 2 for g in d.merge_log)


def test_merge_log_is_off_by_default_and_writes_tsv_when_on(bare_weconvert, tmp_path, monkeypatch):
    monkeypatch.setenv("WEST_SIM_ROOT", str(tmp_path))
    _, Driver = _load_driver(bare_weconvert, tmp_path, monkeypatch, n_split=5)
    _driver(Driver, _ensemble(40))._run_we()
    assert not (tmp_path / "merge_dists.tsv").exists()
    on = tmp_path / "on"
    on.mkdir()
    monkeypatch.setenv("WEST_SIM_ROOT", str(on))
    mod, Driver = _load_driver(bare_weconvert, on, monkeypatch, n_split=5, merge_log=True)
    _driver(Driver, _ensemble(40))._run_we()
    mod._merge_log_fh.close()
    rows = (on / "merge_dists.tsv").read_text().splitlines()
    assert rows[0].split("\t") == ["iter", "euclidean", "w_a", "w_b", "dNANOG", "dGATA6"]
    assert len(rows) == 1 + 5


# ---------------------------------------------------------------------------
# binning_options parsing (style: mabl)
# ---------------------------------------------------------------------------

def _parsed(bare_weconvert, **binning):
    wc = bare_weconvert(dims=2)
    wc._parse_mabl_opts(binning)
    return wc


def test_mabl_option_defaults(bare_weconvert):
    wc = _parsed(bare_weconvert)
    assert wc.log_scale == [False, False]
    assert (wc.merge_progress_floor, wc.local_merge, wc.merge_pool, wc.merge_log) == (0.01, False, 4, False)
    assert _parsed(bare_weconvert, log_scale=True, start=[1, 1], target=[5, 5]).log_scale == [True, True]


@pytest.mark.parametrize("binning", [
    dict(log_scale=[True]),                                  # wrong length
    dict(coord_weights=[1, 1, 1]),                           # wrong length
    dict(log_scale=[True, False], start=[0, 0]),             # log needs start > 0
    dict(merge_pool=1),
    dict(merge_progress_floor=-1),
])
def test_mabl_option_errors(bare_weconvert, binning):
    with pytest.raises(SystemExit):
        _parsed(bare_weconvert, **binning)


@pytest.mark.parametrize("local", [False, True])
def test_walkers_split_this_iteration_are_not_merged(bare_weconvert, tmp_path, monkeypatch, local):
    """A split's children (same point, equal weights) would pair up by rank and by distance
    alike; merging them would undo the split, so they are not merge candidates."""
    # no coordinate scored: every walker has zero progress, so merge score is 1 / (W * floor)
    # and the half-weight children are the top merge candidates; each child's nearest
    # neighbour is its sibling (distance 0), so local merging would pick it too
    _, Driver = _load_driver(bare_weconvert, tmp_path, monkeypatch, n_split=5, local_merge=local,
                             coord_weights=[0, 0])
    d = _driver(Driver, _ensemble(40))
    children = []
    split = Driver._split_walker

    def tracking_split(self, segment, m, bin):
        out = split(self, segment, m, bin)
        children.extend(out)
        return out
    monkeypatch.setattr(Driver, "_split_walker", tracking_split)
    d._run_we()
    merged = {id(w) for group in d.merge_log for w in group}
    assert len(children) == 10 and not merged & {id(c) for c in children}
    assert _totals(d) == (40, pytest.approx(1.0))


def test_weight_cap_children_are_not_merged_back(bare_weconvert, tmp_path, monkeypatch):
    # the heavy walker sits at the start (zero progress) and the light ones at the target,
    # so the cap's children have the highest merge scores
    _, Driver = _load_driver(bare_weconvert, tmp_path, monkeypatch, n_split=0, weight_cap_factor=2.0)
    walkers = [Walker(0.5 / 9, [[10, 10], [10, 10]]) for _ in range(9)]
    heavy = Walker(0.5, [[0, 0], [0, 0]])
    d = _driver(Driver, walkers + [heavy])
    d._run_we()
    merged = [w for group in d.merge_log for w in group]
    assert all(m in walkers for m in merged)          # only the original light walkers were merged
