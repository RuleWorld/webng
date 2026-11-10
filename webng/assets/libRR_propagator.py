from __future__ import division, print_function; __metaclass__ = type
import numpy as np
import westpa, copy, time, random, os
from westpa.core.propagators import WESTPropagator
from westpa.core.segment import Segment
import roadrunner as librr
import logging
log = logging.getLogger(__name__)
log.debug('loading module %r' % __name__)


class librrPropagator(WESTPropagator):
    def __init__(self, rc=None):
        super(librrPropagator, self).__init__(rc)
        config = self.rc.config
        for key in [('west','librr','init','model_file'),
                    ('west','librr','init','init_time_step'),
                    ('west','librr','init','final_time_step'),
                    ('west','librr','init','num_time_step'),
                    ('west','librr','data','pcoords')]:
            config.require(key)
        self.runner_config = {}
        self.runner_config['model_file']  = os.path.normpath(
            config['west','librr','init','model_file'])
        self.runner_config['init_ts']     = config['west','librr','init','init_time_step']
        self.runner_config['final_ts']    = config['west','librr','init','final_time_step']
        self.runner_config['num_ts']      = config['west','librr','init','num_time_step']
        self.runner_config['pcoord_keys'] = config['west','librr','data','pcoords']
        self.runner = librr.RoadRunner(self.runner_config['model_file'])
        self.runner.setIntegrator('gillespie')
        self.runner.setIntegratorSetting('gillespie', 'variable_step_size', False)
        self.runner.setIntegratorSetting('gillespie', 'nonnegative', True)
        self.initial_pcoord  = self.get_initial_pcoords()
        self.full_state_keys = self.get_full_state_keys()
        self.runner.timeCourseSelections = self.runner_config['pcoord_keys']

    def get_pcoord(self, state):
        state.pcoord = copy.copy(self.initial_pcoord)

    def gen_istate(self, basis_state, initial_state):
        # Sample a microstate from the pool and set both the
        # runner state and the recorded pcoord so WESTPA's bin
        # assignment reflects where the walker actually starts.
        pool_path = os.path.join(
            os.environ.get('WEST_SIM_ROOT', '.'), 'bstates',
            'microstate_pool.dat')
        if os.path.isfile(pool_path):
            data = np.loadtxt(pool_path, comments='#')
            if data.ndim == 1:
                data = data.reshape(1, -1)
            weights = data[:, 0]
            counts  = data[:, 1:]
            weights = weights / weights.sum()
            idx     = np.random.choice(len(counts), p=weights)
            pool_ms = np.array(counts[idx], dtype=np.float64)
            vec     = self._build_full_state(pool_ms)
            if vec is not None:
                self.runner.resetAll()
                self.set_runner_state(vec)
        initial_state.pcoord = self.get_initial_pcoords()
        return initial_state

    def get_initial_pcoords(self):
        return [self.runner[x] for x in self.runner_config['pcoord_keys']]

    def get_full_state_keys(self):
        fs    = self.runner.getFloatingSpeciesAmountsNamedArray().colnames
        concs = ['[' + x + ']' for x in fs]
        return fs + concs

    def get_final_state(self):
        return [self.runner[x] for x in self.full_state_keys]

    def set_runner_state(self, state):
        if all(x == -1 for x in state):
            self.runner.resetAll()
        else:
            for i, val in enumerate(state):
                self.runner.setValue(self.full_state_keys[i], val)

    def _build_full_state(self, pool_microstate):
        '''
        Convert a pool_microstate vector (raw species amounts, length
        n_species) into the full_state_keys-compatible vector
        (amounts + concentrations, length 2*n_species).
        For SSA models volume=1 so amount == concentration.
        Returns None if length does not match.
        '''
        n_sp = len(self.full_state_keys) // 2
        counts = list(pool_microstate)
        if len(counts) != n_sp:
            log.warning(
                'pool_microstate length %d != expected %d; using resetAll().',
                len(counts), n_sp)
            return None
        return counts + counts   # amounts then concentrations

    def propagate(self, segments):
        for segment in segments:
            piter     = segment.n_iter - 1
            starttime = time.time()
            seed      = random.randint(0, 2**14)
            self.runner.resetAll()
            self.runner.setIntegratorSetting('gillespie', 'seed', seed)

            if piter == 0:
                # ── First iteration ─────────────────────────────
                # RestartDriver does not run at iter 1, so we sample
                # directly from pool_microstate if it was set during
                # w_init (future), otherwise use model defaults.
                pool_ms = segment.data.get('pool_microstate', None)
                if pool_ms is not None:
                    vec = self._build_full_state(pool_ms)
                    if vec is not None:
                        self.set_runner_state(vec)
                        log.debug('iter1 seg %d: set state from pool.',
                                  segment.seg_id)
                # else: resetAll() already called — model defaults

            else:
                restart = segment.data.get('restart_state')
                if restart is not None and not all(x == -1 for x in restart):
                    # ── Normal continuing walker ─────────────────
                    self.set_runner_state(restart)
                else:
                    # ── Recycled walker ──────────────────────────
                    pool_ms = segment.data.get('pool_microstate', None)
                    if pool_ms is not None:
                        vec = self._build_full_state(pool_ms)
                        if vec is not None:
                            self.set_runner_state(vec)
                            log.debug(
                                'recycled seg %d: set state from pool.',
                                segment.seg_id)
                    # else: resetAll() already called — graceful fallback

            result = self.runner.simulate(
                self.runner_config['init_ts'],
                self.runner_config['final_ts'],
                self.runner_config['num_ts'],
            )
            segment.data['final_state'] = self.get_final_state()
            segment.data['seed']        = seed
            segment.pcoord              = result
            segment.data.pop('pool_microstate', None)
            segment.walltime            = time.time() - starttime
            segment.cputime             = 0
            segment.status              = segment.SEG_STATUS_COMPLETE
        return segments