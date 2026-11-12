"""
MABL: a weighted-ensemble driver for WESTPA that resamples in one bin by score
instead of by bin occupancy.

`webng setup` copies this file into the simulation folder when
binning_options.style is "mabl" and fills in the configuration block below
from binning_options and the recycling block of the webng config. The values
in the block here are placeholders (a 2-D example) so the file imports on its
own. After setup you can edit the block in the simulation folder.

Each iteration (after recycling walkers that reached the recycle region):
  1. Weight cap: any walker heavier than MABL_WEIGHT_CAP_FACTOR x the average
     weight is split down.
  2. Progress S_i: a weighted sum over progress coordinates of how far walker
     i has come from MABL_START toward MABL_TARGET, each term in [0, 1],
     linear or (MABL_LOG_SCALE) in fold change.
  3. Splits: the MABL_N_SPLIT walkers with the highest S_i / -log(W_i) are
     split in two.
  4. Merges: as many pairwise merges as walkers were added. Candidates are the
     walkers with the highest merge score 1 / (W_i * max(S_i, floor)), i.e.
     light and unpromising, except walkers created by a split in this
     iteration (merging a split's two children would undo it). Pairs are
     taken by merge-score rank, or (local
     merging) each candidate is merged with its nearest neighbour. Always
     pairwise, so no merge creates a walker heavier than W_a + W_b, and the
     survivor is chosen by weight (WESTPA's _merge_walkers), which keeps the
     resampling statistically unbiased.
"""
import logging
import math
import operator
import os

import numpy as np
from westpa.core.we_driver import WEDriver, NewWeightEntry
from westpa.core.segment import Segment

log = logging.getLogger(__name__)

# >>> MABL configuration (written by webng setup from binning_options and recycling)
# Progress coordinates, in order (west.cfg); the lists below follow this order.
MABL_PCOORDS = ['X', 'Y']

# Start and target microstate per coordinate (used for scoring only).
MABL_START = [0, 0]
MABL_TARGET = [10, 10]

# Per-coordinate importance weights. Zero excludes a coordinate from scoring
# (e.g. one that doesn't change monotonically along the transition).
MABL_COORD_WEIGHTS = [1, 1]

# Per-coordinate progress scale. False: linear, p = 1 - |x - target| / |start - target|.
# True: log (fold change), p = log(x / start) / log(target / start); needs start and
# target > 0. Both clipped to [0, 1]. Log is sensitive at low counts, so it suits a
# coordinate that must rise from a low start; case by case.
MABL_LOG_SCALE = [False, False]

# Walkers split per iteration. Each split adds one walker; an equal number of
# pairwise merges removes them again, so the walker count stays constant.
MABL_N_SPLIT = 5

# Weight cap: safety net only. Any walker heavier than this factor times the
# average weight is split down.
MABL_WEIGHT_CAP_FACTOR = 10.0

# Merge score = 1 / (W * max(S, floor)). Without a floor, walkers with zero progress
# (at or beyond the start on every scored coordinate) would be merged first however
# heavy they are; with it they are ranked by weight like everyone else.
MABL_MERGE_PROGRESS_FLOOR = 0.01

# Local merging. False: merge candidates are paired by merge-score rank (1st with
# 2nd, ...), wherever they are. True: the top MABL_MERGE_POOL x n_merge walkers by
# merge score are candidates; going down the ranking, each unpaired walker is merged
# with the nearest unpaired one among them (Euclidean distance in the progress
# coordinates). Larger pools find closer partners, but partners may then rank lower.
MABL_LOCAL_MERGE = False
MABL_MERGE_POOL = 4

# Debugging: append every pairwise merge (iteration, distance, weights, per-coordinate
# differences) to merge_dists.tsv in the simulation folder.
MABL_MERGE_LOG = False

# Recycling region: [min, max] per coordinate, in pcoord order. None disables that
# bound; None for the whole region disables recycling.
MABL_RECYCLE_REGION = None
# <<< end MABL configuration


_merge_log_fh = None


def _merge_log():
    """merge_dists.tsv in the simulation folder, opened once (header if new)."""
    global _merge_log_fh
    if _merge_log_fh is None or _merge_log_fh.closed:
        path = os.path.join(os.environ.get('WEST_SIM_ROOT', '.'), 'merge_dists.tsv')
        is_new = not os.path.isfile(path)
        _merge_log_fh = open(path, 'a', buffering=1)
        if is_new:
            _merge_log_fh.write('iter\teuclidean\tw_a\tw_b\t'
                                + '\t'.join('d' + str(p) for p in MABL_PCOORDS) + '\n')
    return _merge_log_fh


class MABLDriver(WEDriver):

    def _in_recycle_region(self, pcoord_frame):
        if MABL_RECYCLE_REGION is None:
            return False
        for d, (lo, hi) in enumerate(MABL_RECYCLE_REGION):
            val = pcoord_frame[d]
            if lo is not None and val < lo:
                return False
            if hi is not None and val > hi:
                return False
        return True

    def _recycle_walkers(self):
        """
        Recycle walkers that have reached the target macrostate back to
        basis states, recording NewWeightEntry for each so w_fluxanl
        correctly accounts for probability flux into the target.
        """
        self.new_weights = []
        if MABL_RECYCLE_REGION is None:
            return
        if not self.target_states or not hasattr(self, '_parent_map'):
            return

        target_state    = next(iter(self.target_states.values()))
        istateiter      = iter(self.avail_initial_states.values())
        used_istate_ids = set()
        n_recycled      = 0
        recycled_weight = 0.0

        for bin in self.next_iter_binning:
            for segment in set(bin):
                if not self._in_recycle_region(segment.pcoord[0]):
                    continue
                parent = self._parent_map[segment.parent_id]
                parent.endpoint_type = Segment.SEG_ENDPOINT_RECYCLED
                try:
                    initial_state = next(istateiter)
                except StopIteration:
                    raise RuntimeError(
                        'MABL recycling: ran out of available initial states. '
                        'Increase --segs-per-state in init.sh or reduce walkers.'
                    )
                istate_assignment = self.bin_mapper.assign([initial_state.pcoord])[0]
                segment.parent_id = -(initial_state.state_id + 1)
                segment.pcoord[0] = initial_state.pcoord
                self.new_weights.append(
                    NewWeightEntry(
                        source_type       = NewWeightEntry.NW_SOURCE_RECYCLED,
                        weight            = parent.weight,
                        prev_seg_id       = parent.seg_id,
                        prev_init_pcoord  = parent.pcoord[0].copy(),
                        prev_final_pcoord = parent.pcoord[-1].copy(),
                        new_init_pcoord   = initial_state.pcoord.copy(),
                        target_state_id   = target_state.state_id,
                        initial_state_id  = initial_state.state_id,
                    )
                )
                bin.remove(segment)
                self.next_iter_binning[istate_assignment].add(segment)
                initial_state.iter_used = segment.n_iter
                used_istate_ids.add(initial_state.state_id)
                n_recycled      += 1
                recycled_weight += parent.weight

        for state_id in used_istate_ids:
            self.used_initial_states[state_id] = \
                self.avail_initial_states.pop(state_id)
        log.info('Recycled %d walkers, total weight = %.4e',
                 n_recycled, recycled_weight)

    @property
    def n_istates_needed(self):
        if self.final_binning is None or MABL_RECYCLE_REGION is None:
            return 0
        n_will_recycle = sum(
            1 for b in self.final_binning
            for seg in b
            if self._in_recycle_region(seg.pcoord[-1])
        )
        return max(0, n_will_recycle - len(self.avail_initial_states))

    def _split_into(self, bin, segment, n):
        """Split segment into n children in bin; the children are marked as
        split this iteration, so they are not merged again before they move."""
        bin.remove(segment)
        children = self._split_walker(segment, n, bin)
        self._split_this_iter.update(id(c) for c in children)
        bin.update(children)

    def _split_by_data(self, bin, to_split, split_into):
        for segment in to_split:
            self._split_into(bin, segment, split_into)

    def _mergeable(self, segment):
        """Merge candidates: not just recycled (parent_id < 0) and not created
        by a split this iteration. Merging two children of the same split, which
        sit at the same point with equal weights and so pair up by rank and by
        distance alike, would just undo the split."""
        return segment.parent_id >= 0 and id(segment) not in getattr(self, '_split_this_iter', ())

    def _merge_pair(self, bin, seg_a, seg_b):
        """
        Merge exactly two walkers into one survivor.

        Consistent with standard WESTPA's pairwise merge strategy
        (_adjust_count), which always merges pairs rather than N-into-1.
        The survivor is chosen by weighted random selection within
        _merge_walkers, satisfying Rule 2 of statistically unbiased WE
        resampling (Huber & Kim 1996): the surviving trajectory must be
        chosen according to its weight.

        Maximum weight of any survivor: W_a + W_b, so no merge creates a
        walker that could dominate the ensemble.
        """
        if MABL_MERGE_LOG:
            diff = np.abs(np.asarray(seg_a.pcoord[0], float) - np.asarray(seg_b.pcoord[0], float))
            _merge_log().write('{}\t{:.4f}\t{:.6e}\t{:.6e}\t{}\n'.format(
                seg_a.n_iter, np.linalg.norm(diff), seg_a.weight, seg_b.weight,
                '\t'.join('{:.2f}'.format(x) for x in diff)))
        bin.discard(seg_a)
        bin.discard(seg_b)
        new_segment, parent = self._merge_walkers([seg_a, seg_b], None, bin)
        bin.add(new_segment)

    def _merge_group(self, bin, group):
        """Merge three or more walkers into one (only for an odd leftover)."""
        bin.difference_update(group)
        new_segment, _ = self._merge_walkers(group, None, bin)
        bin.add(new_segment)

    def _rebuild(self, bin):
        """Rebuild segments and pcoord arrays from bin after any split/merge."""
        segments = np.array(
            sorted(bin, key=operator.attrgetter('weight')),
            dtype=np.object_
        )
        all_pcoords = np.array([s.pcoord for s in segments])
        return segments, all_pcoords

    def _compute_progress(self, all_pcoords, ndims):
        """
        Weighted sum progress toward target, from each walker's first frame.

        linear: p_d = max(0, 1 - |x_d - target_d| / |start_d - target_d|)
        log:    p_d = clip(log(x_d / start_d) / log(target_d / start_d), 0, 1)
        S_i = sum_d( w_d * p_d )
        """
        nsegs = all_pcoords.shape[0]
        progresses = np.zeros((nsegs, ndims), dtype=float)
        for d in range(ndims):
            span = abs(MABL_START[d] - MABL_TARGET[d])
            if span == 0:
                progresses[:, d] = 1.0
                continue
            x = np.asarray(all_pcoords[:, 0, d], dtype=float)
            if MABL_LOG_SCALE[d]:
                p = np.log(np.maximum(x, 1.0) / MABL_START[d]) / np.log(MABL_TARGET[d] / MABL_START[d])
                progresses[:, d] = np.clip(p, 0.0, 1.0)
            else:
                progresses[:, d] = np.maximum(0.0, 1.0 - np.abs(x - MABL_TARGET[d]) / span)
        coord_weights = np.array(MABL_COORD_WEIGHTS, dtype=float)
        return np.sum(progresses * coord_weights, axis=1)

    def _compute_split_scores(self, segments, sum_progress):
        """Split score = S_i * (1 / -log(W_i))"""
        weights     = np.array([s.weight for s in segments])
        log_weights = -1.0 * np.log(weights)
        return sum_progress * (1.0 / log_weights)

    def _compute_ghost_scores(self, segments, sum_progress):
        """Merge (ghost) score = 1 / (W_i * max(S_i, MABL_MERGE_PROGRESS_FLOOR))"""
        weights = np.array([s.weight for s in segments])
        return 1.0 / (weights * np.maximum(sum_progress, MABL_MERGE_PROGRESS_FLOOR))

    def _rank_merge_groups(self, segments, ghost_scores, n_merge):
        """
        Merge groups by merge-score rank: the 2 x n_merge highest-scoring
        walkers, 1st with 2nd, 3rd with 4th, ... With an odd count (fewer
        mergeable walkers than needed), the last three form one group, so no
        walker is left unpaired -- and none is merged twice, which would count
        its weight twice.
        """
        selected = []
        for idx in np.argsort(-ghost_scores):
            if len(selected) >= n_merge * 2:
                break
            if not self._mergeable(segments[idx]):
                continue
            selected.append(idx)
        n = len(selected)
        if n % 2 == 1 and n >= 3:
            n_pairs, tail = (n - 3) // 2, [selected[-3:]]
        else:
            n_pairs, tail = n // 2, []
        return [selected[2 * i:2 * i + 2] for i in range(n_pairs)] + tail

    def _local_merge_groups(self, segments, all_pcoords, ghost_scores, n_merge):
        """
        Local merge groups: candidates are the MABL_MERGE_POOL x n_merge
        highest-scoring walkers; going down the ranking, each unpaired walker
        is paired with its nearest unpaired candidate (progress coordinates,
        first frame). An odd leftover joins its nearest group.
        """
        order = [i for i in np.argsort(-ghost_scores) if self._mergeable(segments[i])]
        pool = np.array(order[:max(2, MABL_MERGE_POOL) * n_merge], dtype=int)
        pts = np.asarray(all_pcoords[pool, 0, :], dtype=float)
        free = np.ones(len(pool), dtype=bool)
        groups = []
        for a in range(len(pool)):
            if len(groups) >= n_merge:
                break
            if not free[a]:
                continue
            free[a] = False
            cand = np.nonzero(free)[0]
            if len(cand) == 0:
                if groups:
                    near = np.argmin([np.linalg.norm(pts[g[0]] - pts[a]) for g in groups])
                    groups[near].append(a)
                break
            b = cand[np.argmin(np.linalg.norm(pts[cand] - pts[a], axis=1))]
            free[b] = False
            groups.append([a, b])
        return [[int(pool[i]) for i in g] for g in groups]

    def _is_initialization(self):
        """
        True when no current walker has moved yet, i.e. this resampling call
        comes from w_init's populate_initial, before any propagation. Checks
        every pcoord dimension: a walker has moved if any coordinate differs
        between its first and last frame.
        """
        pcoords = np.array([s.pcoord for s in self.current_iter_segments])
        return not np.any(pcoords[:, 0, :] != pcoords[:, -1, :])

    def _run_we(self):
        self.new_weights = self.new_weights or []
        self._split_this_iter = set()
        self._recycle_walkers()
        self._check_pre()

        for bin in self.next_iter_binning:
            if len(bin) == 0:
                continue

            segments, all_pcoords = self._rebuild(bin)
            nsegs, nframes, ndims = all_pcoords.shape

            # Skip resampling on the initialization iteration
            if self._is_initialization():
                continue

            # ----------------------------------------------------------
            # STEP 1: Weight cap (safety net)
            # ----------------------------------------------------------
            target_weight = 1.0 / nsegs
            max_weight    = MABL_WEIGHT_CAP_FACTOR * target_weight
            n_cap_extra   = 0

            weights    = np.array([s.weight for s in segments])
            cap_needed = np.where(weights > max_weight)[0]

            if len(cap_needed) > 0:
                for idx in cap_needed:
                    seg        = segments[idx]
                    n_children = math.ceil(seg.weight / max_weight)
                    self._split_into(bin, seg, n_children)
                    n_cap_extra += n_children - 1
                    log.warning('cap: w=%.4e -> %d children',
                                seg.weight, n_children)
                segments, all_pcoords = self._rebuild(bin)
                nsegs = len(segments)
                log.warning('cap: %d walkers capped, %d extra, total=%d',
                            len(cap_needed), n_cap_extra, nsegs)

            # ----------------------------------------------------------
            # STEP 2: Compute weighted sum progress S_i
            # ----------------------------------------------------------
            sum_progress = self._compute_progress(all_pcoords, ndims)

            # ----------------------------------------------------------
            # STEP 3: MABL splits
            # ----------------------------------------------------------
            split_scores = self._compute_split_scores(segments, sum_progress)

            to_split_idx = []
            for idx in np.argsort(-split_scores):
                if len(to_split_idx) >= MABL_N_SPLIT:
                    break
                if segments[idx].parent_id < 0:
                    continue
                to_split_idx.append(idx)

            if to_split_idx:
                self._split_by_data(bin, list(segments[to_split_idx]), 2)

            # Rebuild and recompute after splits
            segments, all_pcoords = self._rebuild(bin)
            nsegs        = len(segments)
            sum_progress = self._compute_progress(all_pcoords, ndims)

            # ----------------------------------------------------------
            # STEP 4: Pairwise merging (merge-score candidates)
            # ----------------------------------------------------------
            ghost_scores = self._compute_ghost_scores(segments, sum_progress)
            n_merge      = n_cap_extra + len(to_split_idx)
            if n_merge == 0:
                continue
            if MABL_LOCAL_MERGE:
                groups = self._local_merge_groups(segments, all_pcoords, ghost_scores, n_merge)
            else:
                groups = self._rank_merge_groups(segments, ghost_scores, n_merge)
            for group in groups:
                if len(group) == 2:
                    self._merge_pair(bin, segments[group[0]], segments[group[1]])
                else:
                    self._merge_group(bin, list(segments[group]))
            if sum(len(g) - 1 for g in groups) < n_merge:
                log.warning('MABL merge: removed %d of %d added walkers (too few mergeable walkers)',
                            sum(len(g) - 1 for g in groups), n_merge)

        self._check_post()
        log.debug('used initial states: {!r}'.format(self.used_initial_states))
        log.debug('available initial states: {!r}'.format(self.avail_initial_states))
