from __future__ import division, print_function; __metaclass__ = type
import numpy as np
import ctypes, os, time, random, logging
from westpa.core.propagators import WESTPropagator
log = logging.getLogger(__name__)
log.debug('loading module %r' % __name__)


def _load_bng_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(here, 'libssa.so')
    if not os.path.isfile(lib_path):
        raise FileNotFoundError(
            'libssa.so not found at {}. It should have been copied '
            'into the simulation root by \"webng setup\".'.format(lib_path)
        )
    try:
        lib = ctypes.CDLL(lib_path)
    except OSError as e:
        raise OSError(
            'Found libssa.so at {} but could not load it ({}). '
            'This is almost always a platform/architecture mismatch '
            '-- e.g. a Linux-built libssa.so on macOS or Windows. '
            'Rebuild it for this machine with build_libssa.sh.'
            .format(lib_path, e)
        ) from e

    lib.ssa_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.ssa_init.restype  = ctypes.c_int

    lib.ssa_set_state.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.ssa_set_state.restype  = ctypes.c_int

    lib.ssa_run.argtypes = [ctypes.c_double, ctypes.c_double]
    lib.ssa_run.restype  = ctypes.c_int

    lib.ssa_run_multi.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ]
    lib.ssa_run_multi.restype = ctypes.c_int

    lib.ssa_get_state.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.ssa_get_state.restype  = ctypes.c_int

    lib.ssa_get_n_species.argtypes = []
    lib.ssa_get_n_species.restype  = ctypes.c_int

    lib.ssa_get_last_error.argtypes = []
    lib.ssa_get_last_error.restype  = ctypes.c_char_p

    lib.ssa_reseed.argtypes = [ctypes.c_int]
    lib.ssa_reseed.restype  = None
    return lib


def _parse_net_groups(net_file):
    '''
    Parse the 'begin groups' block of a .net file into
    {observable_name: [(0-based species index, coefficient), ...]}.
    Handles both bare indices ('3') and weighted terms ('2*3'),
    and sums over every term rather than assuming a single species.
    '''
    groups, in_block = {}, False
    with open(net_file) as f:
        for line in f:
            s = line.strip()
            if s.startswith('begin groups'):
                in_block = True
                continue
            if s.startswith('end groups'):
                break
            if in_block and s and not s.startswith('#'):
                parts = s.split()
                if len(parts) >= 3:
                    name  = parts[1]
                    terms = []
                    for token in parts[2].split(','):
                        token = token.strip()
                        if '*' in token:
                            coeff_str, idx_str = token.split('*', 1)
                            coeff = float(coeff_str)
                        else:
                            idx_str, coeff = token, 1.0
                        try:
                            terms.append((int(idx_str) - 1, coeff))
                        except ValueError:
                            pass
                    if terms:
                        groups[name] = terms
    return groups


def _parse_net_species_counts(net_file, n_species):
    '''Return a length-n_species array of species counts from a .net file.'''
    counts = np.zeros(n_species, dtype=np.float64)
    in_block = False
    with open(net_file) as f:
        for line in f:
            s = line.strip()
            if s.startswith('begin species'):
                in_block = True
                continue
            if s.startswith('end species'):
                break
            if in_block and s and not s.startswith('#'):
                parts = s.split()
                if len(parts) >= 3:
                    idx = int(parts[0]) - 1
                    if idx >= n_species:
                        continue
                    try:
                        counts[idx] = float(parts[2])
                    except ValueError:
                        raise ValueError(
                            'species {} in {} has a non-numeric count {!r}; '
                            'starting counts must be numbers.'
                            .format(idx + 1, net_file, parts[2])
                        )
    return counts


class bngPropagator(WESTPropagator):
    def __init__(self, rc=None):
        super(bngPropagator, self).__init__(rc)
        config = self.rc.config
        for key in [('west', 'bng', 'init', 'net_file'),
                    ('west', 'bng', 'init', 'init_time_step'),
                    ('west', 'bng', 'init', 'final_time_step'),
                    ('west', 'bng', 'init', 'num_time_step'),
                    ('west', 'bng', 'data', 'pcoords')]:
            config.require(key)

        self.net_file    = os.path.normpath(config['west', 'bng', 'init', 'net_file'])
        self.t_start     = float(config['west', 'bng', 'init', 'init_time_step'])
        self.t_end       = float(config['west', 'bng', 'init', 'final_time_step'])
        self.num_ts      = int(config['west', 'bng', 'init', 'num_time_step'])
        self.pcoord_keys = config['west', 'bng', 'data', 'pcoords']
        self.gillespie_update_interval = int(
            config.get(('west', 'bng', 'init', 'gillespie_update_interval'), 1)
        )

        self._lib = _load_bng_lib()
        seed = random.randint(0, 2**14)
        status = self._lib.ssa_init(
            self.net_file.encode(), seed, self.gillespie_update_interval
        )
        if status != 0:
            err = self._lib.ssa_get_last_error().decode()
            raise RuntimeError('bngPropagator: ssa_init failed: {}'.format(err))

        self._n_species = self._lib.ssa_get_n_species()
        log.info('bngPropagator initialized: %d species, net_file=%s',
                 self._n_species, self.net_file)

        # Observable definitions: name -> [(species_idx, coeff), ...].
        # Every term in the group is summed -- this is NOT just the
        # first listed species, so multi-species / weighted
        # observables (e.g. 'Group(2) A + Group(3) B') come out correct.
        group_map = _parse_net_groups(self.net_file)
        self._pcoord_terms = []
        for key in self.pcoord_keys:
            if key in group_map:
                self._pcoord_terms.append(group_map[key])
            else:
                log.warning(
                    'pcoord key %r not found as an observable in %s; '
                    'treating it as a bare 1-based species index.',
                    key, self.net_file)
                self._pcoord_terms.append([(int(key) - 1, 1.0)])

        self._init_counts = _parse_net_species_counts(self.net_file, self._n_species)
        self._initial_pcoord = self._weighted_pcoord(self._init_counts)

        # Pre-allocate buffers
        self._multi_buf = np.zeros(self.num_ts * self._n_species, dtype=np.float64)
        self._state_buf = np.zeros(self._n_species, dtype=np.float64)

    def _weighted_pcoord(self, state):
        '''Each observable is the weighted sum of its species terms.'''
        out = np.zeros(len(self._pcoord_terms), dtype=np.float32)
        for j, terms in enumerate(self._pcoord_terms):
            out[j] = sum(coeff * state[idx] for idx, coeff in terms)
        return out

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

    def _bstate_net_path_for_segment(self, segment):
        '''
        The .net file of the basis state a new trajectory started from.
        WESTPA fills self.initial_states / self.basis_states (via
        update_basis_initial_states) before every propagate() call, for
        iteration-1 walkers and recycled walkers alike, so no side
        channel is needed. Returns None if the lookup comes up empty.
        '''
        parent_id = getattr(segment, 'parent_id', None)
        if parent_id is None or parent_id >= 0:
            return None
        istate_id = -(parent_id + 1)
        initial_state = self.initial_states.get(istate_id)
        if initial_state is None:
            log.warning('no initial_state for istate_id=%d (parent_id=%d)',
                        istate_id, parent_id)
            return None
        basis_state = self.basis_states.get(initial_state.basis_state_id)
        if basis_state is None:
            log.warning('no basis_state for basis_state_id=%d',
                        initial_state.basis_state_id)
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
        Never "whatever the simulator was left in": that would chain
        this segment onto the previous one run in the same worker.
        '''
        parent_id = getattr(segment, 'parent_id', None)
        if parent_id is not None and parent_id >= 0:
            restart = segment.data.get('restart_state')
            if restart is not None and not all(x == -1 for x in restart):
                return np.array(restart, dtype=np.float64)
            raise RuntimeError(
                'segment {} continues from parent {} but has no restart_state; '
                'is restart_plugin.RestartDriver enabled in west.cfg?'
                .format(getattr(segment, 'seg_id', '?'), parent_id)
            )

        pool_ms = segment.data.get('pool_microstate', None)
        if pool_ms is not None:
            state = np.array(pool_ms, dtype=np.float64)
            if len(state) != self._n_species:
                log.warning('pool_microstate length %d != expected %d; '
                            'zero-padding/truncating.', len(state), self._n_species)
                full = np.zeros(self._n_species, dtype=np.float64)
                m = min(len(state), self._n_species)
                full[:m] = state[:m]
                state = full
            return state

        net_path = self._bstate_net_path_for_segment(segment)
        if net_path:
            return _parse_net_species_counts(net_path, self._n_species)
        log.warning('segment %s: no basis state found, starting from %s',
                    getattr(segment, 'seg_id', '?'), self.net_file)
        return self._init_counts.copy()

    def get_pcoord(self, state):
        # state is a BasisState here; its auxref names its .net file
        net_path = self._resolve_net_path(getattr(state, 'auxref', None))
        if net_path:
            counts = _parse_net_species_counts(net_path, self._n_species)
            state.pcoord = self._weighted_pcoord(counts)
        else:
            state.pcoord = self._initial_pcoord.copy()

    def gen_istate(self, basis_state, initial_state):
        net_path = self._resolve_net_path(getattr(basis_state, 'auxref', None))
        if net_path:
            counts = _parse_net_species_counts(net_path, self._n_species)
            initial_state.pcoord = self._weighted_pcoord(counts)
        else:
            initial_state.pcoord = self._initial_pcoord.copy()
        return initial_state

    def propagate(self, segments):
        for segment in segments:
            starttime = time.time()
            seed  = random.randint(0, 2**14)

            start = self._start_state(segment)
            ptr = start.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            status = self._lib.ssa_set_state(ptr, self._n_species)
            if status != 0:
                err = self._lib.ssa_get_last_error().decode()
                raise RuntimeError('ssa_set_state returned {}: {}'.format(status, err))
            self._lib.ssa_reseed(seed)

            if self.num_ts == 2:
                status = self._lib.ssa_run(self.t_start, self.t_end)
                if status != 0:
                    err = self._lib.ssa_get_last_error().decode()
                    log.warning('ssa_run returned %d: %s', status, err)

                ptr = self._state_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                self._lib.ssa_get_state(ptr, self._n_species)

                pcoord = np.zeros((2, len(self.pcoord_keys)), dtype=np.float32)
                pcoord[0] = self._weighted_pcoord(start)
                pcoord[1] = self._weighted_pcoord(self._state_buf)
            else:
                out_ptr = self._multi_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                status = self._lib.ssa_run_multi(
                    self.t_start, self.t_end, self.num_ts, out_ptr, self._n_species
                )
                if status != 0:
                    err = self._lib.ssa_get_last_error().decode()
                    log.warning('ssa_run_multi returned %d: %s', status, err)

                all_states = self._multi_buf.reshape(self.num_ts, self._n_species)
                pcoord = np.zeros((self.num_ts, len(self.pcoord_keys)), dtype=np.float32)
                for ts in range(self.num_ts):
                    pcoord[ts] = self._weighted_pcoord(all_states[ts])
                self._state_buf[:] = all_states[-1]

            segment.data['final_state'] = self._state_buf.copy()
            segment.data['seed']        = seed
            segment.data.pop('pool_microstate', None)
            segment.pcoord   = pcoord
            segment.walltime = time.time() - starttime
            segment.cputime  = 0
            segment.status   = segment.SEG_STATUS_COMPLETE
        return segments