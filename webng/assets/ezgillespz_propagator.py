"""
WESTPA propagator that runs a BioNetGen network with ezgillespz: exact Gillespie
SSA on every CPU thread or on a GPU (CUDA on NVIDIA, Metal on Apple silicon).

`webng setup` copies this file into the simulation folder when
propagator_options.propagator_type is "ezgillespz". Its settings come from
west.cfg, section west.ezgillespz:

  init:
    net_file:        ./bngl_conf/init.net
    init_time_step:  0
    final_time_step: <tau>
    num_time_step:   <pcoord_length + 1>   # points per segment, both ends included
    device:          cpu       # cpu | gpu (CUDA or Metal, whichever this machine has) | cuda | metal
    kind:            compact   # ezgillespz engine; see ezgillespz.make_engine
    headroom_bits:   8         # fixed-point headroom above the largest total rate seen
    thin:            false     # true: rate constants exact (ezgillespz thinning), not rounded to the scale
    seed:            null      # an integer makes the run reproducible; null draws a seed from the OS
  data:
    pcoords: [...]             # observables (groups in net_file), or 1-based species indices

Batching. Every segment WESTPA passes to one propagate() call runs as one
batch: one engine launch per progress-coordinate interval, all walkers at
once. west.cfg's propagation.block_size therefore sets the batch size. WEBNG
writes a block size larger than any iteration and runs w_run with the serial
work manager, so each iteration is a single batch in a single process (one
GPU context, nothing pickled between processes).

Start states follow bng_propagator.py: a continuing walker starts from its
parent's final state (restart_plugin.RestartDriver puts it in
segment.data['restart_state']); a new trajectory (iteration 1, or a recycled
walker: parent_id < 0) starts from a microstate drawn from the recycling pool
if there is one, otherwise from its own basis state's .net file.

Fixed point. ezgillespz keeps propensities as 64-bit fixed-point integers.
The scale is set from the walkers' own total rates on the first batch, with
headroom_bits to spare. If a launch's total rate overflows the range, the
scale is made coarser and the launch is repeated from the same start; if it
comes within 2 bits of the limit, the scale is made coarser before the next
launch. This is what ezgillespz's own WeightedEnsemble does.

Seeds. Each batch draws one base seed; launch k of the batch uses
seeds_for(num_time_step - 1, base)[k], and ezgillespz derives each walker's
stream from that seed and the walker's row. segment.data records 'seed' (the
base seed) and 'seed_row' (the row), which reproduce the batch, plus 'events'
(SSA events the segment fired).
"""
from __future__ import division, print_function

import logging
import math
import os
import subprocess
import time

import numpy as np
from westpa.core.propagators import WESTPropagator

try:
    import ezgillespz as gg
    from ezgillespz.network import seeds_for
    from ezgillespz import species_kernel as _sk
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError(
        "propagator_type 'ezgillespz' needs the ezgillespz package in the "
        "environment that runs w_run (e.g. pip install -e /path/to/ezgillespz): "
        "{}".format(e)
    ) from e

log = logging.getLogger(__name__)

MAX_COUNT = 2 ** 31 - 1          # ezgillespz keeps species counts as int32
MAX_HEADROOM_BITS = 44           # beyond this a rate law is effectively unbounded
STAT_CHUNK = 4096                # rows per propensity evaluation when sizing the scale


def ezgillespz_provenance():
    """'ezgillespz <version> (<git commit>) at <path>', for the run log."""
    path = os.path.dirname(os.path.abspath(gg.__file__))
    commit = "commit unknown"
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            commit = "commit " + out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "ezgillespz {} ({}) at {}".format(getattr(gg, "__version__", "?"), commit, path)


def pcoord_weights(model, keys):
    """(len(keys), n_species) weights: an observable (group) of the .net file,
    every species term summed with its coefficient, or a bare 1-based species
    index as bng_propagator.py accepts."""
    W = np.zeros((len(keys), model.ns), dtype=np.float64)
    for i, key in enumerate(keys):
        if key in model.groups:
            W[i] = model.groups[key]
            continue
        try:
            idx = int(key) - 1
        except (TypeError, ValueError):
            raise ValueError(
                "pcoord {!r} is not an observable of the network; observables: {}"
                .format(key, sorted(model.groups))) from None
        if not 0 <= idx < model.ns:
            raise ValueError("pcoord {!r}: species index out of range 1..{}".format(key, model.ns))
        log.warning("pcoord %r is not an observable; using it as a 1-based species index.", key)
        W[i, idx] = 1.0
    return W


def as_counts(state, n_species, what):
    """A start state as int32 counts, refusing what int32 cannot hold."""
    x = np.rint(np.asarray(state, dtype=np.float64))
    if x.shape != (n_species,):
        raise ValueError("{}: expected {} species, got shape {}".format(what, n_species, x.shape))
    if not np.isfinite(x).all() or (x < 0).any() or (x > MAX_COUNT).any():
        raise ValueError("{}: species counts must be finite, >= 0 and below 2^31".format(what))
    return x.astype(np.int32)


class Runner:
    """One ezgillespz engine with its fixed-point scale managed for a long run.

    run(X, seed, tau) advances every row of X by tau and returns the final
    counts and the events per walker. The scale is set on the first call from
    the walkers' total rates; it is made coarser (by 4 bits of headroom) when
    a launch overflows, which repeats the launch, or comes within 2 bits of the
    limit, which applies from the next launch."""

    def __init__(self, model, device="cpu", kind="compact", headroom_bits=8, thin=False):
        self.model, self.device, self.kind = model, device, kind
        self.headroom = int(headroom_bits)
        self.thin = thin
        self.engine = None
        self.peak = 0.0            # largest total rate seen (start states, and peaks inside launches)
        self.rescales = 0

    # -- scale ---------------------------------------------------------------
    def _start_peak(self, X):
        m, peak = self.model, 0.0
        for i in range(0, len(X), STAT_CHUNK):
            chunk = X[i:i + STAT_CHUNK]
            m.note_states(chunk)                  # sizes function tables (functional rate laws)
            a = m.propensities(chunk).sum(1).max()
            if not np.isfinite(a):
                raise ValueError("a rate law is infinite at one of the start states")
            peak = max(peak, float(a))
        return peak

    def _set_scale(self):
        m = self.model
        lp = int(math.ceil(math.log2(max(self.peak, 1e-300))))
        kmax = float(m.rates.max()) if m.nr else 0.0
        bmax = 61 - int(math.floor(math.log2(kmax))) if kmax > 0 else 10 ** 6   # keep every constant below 2^62
        bits = min(62 - self.headroom - lp, bmax)
        while True:
            try:
                m.set_scale(bits, rounding="ignore" if self.thin else None)
                break
            except ValueError:                    # too coarse for the smallest constant
                bits += 1
                if bits > min(62 - lp, bmax):
                    raise
        kw = {"thin": self.thin} if self.thin else {}
        self.engine = gg.make_engine(m, kind=self.kind, device=self.device, **kw)

    def _rescale(self, why):
        self.headroom += 4
        self.rescales += 1
        if self.headroom > MAX_HEADROOM_BITS:
            raise RuntimeError("ezgillespz: the total rate keeps leaving the fixed-point range "
                               "({}); a rate law may be unbounded".format(why))
        log.info("ezgillespz: %s; headroom now %d bits", why, self.headroom)
        self._set_scale()

    def prepare(self, X):
        if self.engine is None:
            if self.device != "cpu" and not gg.gpu_available():
                raise RuntimeError(
                    "ezgillespz: device {!r} was requested but no GPU is usable here. "
                    "Set device: cpu in west.cfg (west.ezgillespz.init) to run on the CPU."
                    .format(self.device))
            self.peak = max(self.peak, self._start_peak(X))
            self._set_scale()

    # -- launches ------------------------------------------------------------
    @staticmethod
    def _flag(out):
        if out.get("flag") is not None:          # compact and rule-level engines
            return out["flag"]
        ev = np.asarray(out["events"])           # the others mark failures in the event counts
        if (ev < 0).any():
            return "badsel" if (ev == _sk.BADSEL).any() else "overflow"
        return None

    def run(self, X, seed, tau):
        X = np.ascontiguousarray(X, dtype=np.int32)
        self.prepare(X)
        while True:
            out = self.engine.segment(X, seed=seed, tau=tau)
            flag = self._flag(out)
            if flag is None:
                break
            if flag == "overflow":
                self._rescale("a launch's total rate overflowed the fixed-point range; repeating it")
                continue
            raise RuntimeError("ezgillespz {} engine flagged walkers ({!r})".format(self.kind, flag))
        scale = 2.0 ** self.model.scale_bits
        peak = float(np.max(out["max_total"])) / scale
        self.peak = max(self.peak, peak)
        if peak * 4.0 > 2.0 ** 62 / scale:          # within two bits of the range
            self._rescale("total rate came within 2 bits of the fixed-point limit")
        return np.asarray(out["X"], dtype=np.int32), np.asarray(out["events"], dtype=np.int64)


class ezgillespzPropagator(WESTPropagator):
    SECTION = ("west", "ezgillespz")

    def __init__(self, rc=None):
        super(ezgillespzPropagator, self).__init__(rc)
        config = self.rc.config
        S = self.SECTION
        for key in [S + ("init", "net_file"), S + ("init", "init_time_step"),
                    S + ("init", "final_time_step"), S + ("init", "num_time_step"),
                    S + ("data", "pcoords")]:
            config.require(key)

        def opt(name, default):
            return config.get(S + ("init", name), default)

        self.net_file = os.path.normpath(config[S + ("init", "net_file")])
        self.t_start = float(config[S + ("init", "init_time_step")])
        self.t_end = float(config[S + ("init", "final_time_step")])
        self.num_ts = int(config[S + ("init", "num_time_step")])
        self.pcoord_keys = list(config[S + ("data", "pcoords")])
        device = str(opt("device", "cpu")).lower()
        if device not in ("cpu", "gpu", "cuda", "metal"):
            raise ValueError("west.ezgillespz.init.device must be cpu, gpu, cuda or metal, got {!r}".format(device))
        seed = opt("seed", None)
        self._setup(device=device, kind=opt("kind", "compact"),
                    headroom_bits=int(opt("headroom_bits", 8)), thin=opt("thin", False),
                    seed=None if seed is None else int(seed))

    def _setup(self, device, kind, headroom_bits, thin, seed):
        """Everything after reading west.cfg; tests call it directly."""
        if self.num_ts < 2:
            raise ValueError("num_time_step must be at least 2 (both ends of the segment)")
        self.tau = self.t_end - self.t_start
        if not self.tau > 0:
            raise ValueError("final_time_step must be greater than init_time_step")
        self._model = gg.load_net(self.net_file)
        self._n_species = self._model.ns
        self._W = pcoord_weights(self._model, self.pcoord_keys)
        self._init_counts = as_counts(self._model.x0, self._n_species, self.net_file)
        self._initial_pcoord = self._pcoords(self._init_counts[None])[0]
        self._bstate_counts = {}
        self._rng = np.random.default_rng(seed)
        # Engine set-up waits for the first propagate(): with a forking work
        # manager it then happens in the worker, since a CUDA context made
        # before fork() is unusable in the child.
        self._runner = Runner(self._model, device=device, kind=kind,
                              headroom_bits=headroom_bits, thin=thin)
        log.info("ezgillespzPropagator: %s; %d species, %d reactions, device=%s, kind=%s, "
                 "headroom=%d bits, thin=%s, seed=%s, net_file=%s",
                 ezgillespz_provenance(), self._n_species, self._model.nr, device, kind,
                 headroom_bits, thin, seed, self.net_file)

    # -- helpers ---------------------------------------------------------------
    def _pcoords(self, X):
        return (np.asarray(X, dtype=np.float64) @ self._W.T).astype(np.float32)

    def _resolve_net_path(self, auxref):
        '''Find the .net file a basis state's auxref names (e.g. 0.net, SC.net).'''
        if not auxref:
            return None
        if isinstance(auxref, bytes):
            auxref = auxref.decode('utf-8').strip()
        if not auxref:
            return None
        net_dir = os.path.dirname(os.path.abspath(self.net_file))
        for candidate in [
            os.path.join(net_dir, os.path.basename(auxref)),
            os.path.abspath(auxref),
            os.path.join('bstates', os.path.basename(auxref)),
            auxref,
        ]:
            if os.path.isfile(candidate):
                return candidate
        log.warning('Could not resolve net file for auxref: %r', auxref)
        return None

    def _net_counts(self, net_path):
        '''Species counts of a basis state's .net file (parsed once, then cached).'''
        counts = self._bstate_counts.get(net_path)
        if counts is None:
            model = gg.load_net(net_path)
            if model.ns != self._n_species:
                raise ValueError("{} has {} species, but {} has {}; basis states must come from "
                                 "the same network".format(net_path, model.ns, self.net_file, self._n_species))
            counts = self._bstate_counts[net_path] = as_counts(model.x0, self._n_species, net_path)
        return counts

    def _state_counts(self, state):
        net_path = self._resolve_net_path(getattr(state, 'auxref', None))
        return self._net_counts(net_path) if net_path else self._init_counts

    def _bstate_net_path_for_segment(self, segment):
        '''
        The .net file of the basis state a new trajectory started from.
        WESTPA fills self.initial_states / self.basis_states before every
        propagate() call, for iteration-1 walkers and recycled walkers alike.
        Returns None if the lookup comes up empty.
        '''
        parent_id = getattr(segment, 'parent_id', None)
        if parent_id is None or parent_id >= 0:
            return None
        istate_id = -(parent_id + 1)
        initial_state = self.initial_states.get(istate_id)
        if initial_state is None:
            log.warning('no initial_state for istate_id=%d (parent_id=%d)', istate_id, parent_id)
            return None
        basis_state = self.basis_states.get(initial_state.basis_state_id)
        if basis_state is None:
            log.warning('no basis_state for basis_state_id=%d', initial_state.basis_state_id)
            return None
        return self._resolve_net_path(getattr(basis_state, 'auxref', None))

    def _start_state(self, segment):
        '''
        Species counts a segment starts from:
          continuing walker (parent_id >= 0): the parent's final state,
            put in segment.data['restart_state'] by the restart plugin
          new trajectory (parent_id < 0: iteration 1, or recycled):
            a microstate the restart plugin drew from the pool, if any;
            otherwise its basis state's .net file; otherwise init.net
        '''
        seg_id = getattr(segment, 'seg_id', '?')
        parent_id = getattr(segment, 'parent_id', None)
        if parent_id is not None and parent_id >= 0:
            restart = segment.data.get('restart_state')
            if restart is not None and not all(x == -1 for x in restart):
                return as_counts(restart, self._n_species, 'segment {} restart_state'.format(seg_id))
            raise RuntimeError(
                'segment {} continues from parent {} but has no restart_state; '
                'is restart_plugin.RestartDriver enabled in west.cfg?'.format(seg_id, parent_id))

        pool_ms = segment.data.get('pool_microstate', None)
        if pool_ms is not None:
            state = np.asarray(pool_ms, dtype=np.float64)
            if len(state) != self._n_species:
                log.warning('pool_microstate length %d != expected %d; zero-padding/truncating.',
                            len(state), self._n_species)
                full = np.zeros(self._n_species, dtype=np.float64)
                m = min(len(state), self._n_species)
                full[:m] = state[:m]
                state = full
            return as_counts(state, self._n_species, 'segment {} pool_microstate'.format(seg_id))

        net_path = self._bstate_net_path_for_segment(segment)
        if net_path:
            return self._net_counts(net_path)
        log.warning('segment %s: no basis state found, starting from %s', seg_id, self.net_file)
        return self._init_counts

    # -- WESTPA interface ------------------------------------------------------
    def get_pcoord(self, state):
        # state is a BasisState here; its auxref names its .net file
        state.pcoord = self._pcoords(self._state_counts(state)[None])[0]

    def gen_istate(self, basis_state, initial_state):
        initial_state.pcoord = self._pcoords(self._state_counts(basis_state)[None])[0]
        return initial_state

    def propagate(self, segments):
        segments = list(segments)
        n = len(segments)
        if n == 0:
            return segments
        starttime = time.time()
        X = np.empty((n, self._n_species), dtype=np.int32)
        for i, segment in enumerate(segments):
            X[i] = self._start_state(segment)

        base_seed = int(self._rng.integers(1, 2 ** 62))
        launch_seeds = seeds_for(self.num_ts - 1, base_seed)
        dt = self.tau / (self.num_ts - 1)
        pcoord = np.empty((n, self.num_ts, len(self.pcoord_keys)), dtype=np.float32)
        pcoord[:, 0] = self._pcoords(X)
        events = np.zeros(n, dtype=np.int64)
        for k in range(1, self.num_ts):
            X, ev = self._runner.run(X, int(launch_seeds[k - 1]), dt)
            events += ev
            pcoord[:, k] = self._pcoords(X)

        # One launch covers the whole batch, so a per-segment time can't be
        # separated: each segment gets an equal share.
        walltime = (time.time() - starttime) / n
        for i, segment in enumerate(segments):
            segment.data['final_state'] = X[i].astype(np.float64)
            segment.data['seed'] = base_seed
            segment.data['seed_row'] = i
            segment.data['events'] = int(events[i])
            segment.data.pop('pool_microstate', None)
            segment.pcoord = pcoord[i]
            segment.walltime = walltime
            segment.cputime = 0
            segment.status = segment.SEG_STATUS_COMPLETE
        return segments
