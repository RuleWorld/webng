from __future__ import division; __metaclass__ = type
import logging, os, random
import numpy as np
log = logging.getLogger(__name__)


def _load_pool(bstates_dir):
    '''
    Load microstate_pool.dat into a numpy array of shape (N, n_species).
    Also returns the weight array of shape (N,) normalised to sum=1.
    Returns (None, None) if the file does not exist (no recycling).
    '''
    pool_path = os.path.join(bstates_dir, 'microstate_pool.dat')
    if not os.path.isfile(pool_path):
        return None, None
    data = np.loadtxt(pool_path, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    weights  = data[:, 0]
    counts   = data[:, 1:]
    weights  = weights / weights.sum()
    return counts, weights


class RestartDriver(object):
    def __init__(self, sim_manager, plugin_config):
        super(RestartDriver, self).__init__()
        if not sim_manager.work_manager.is_master:
            return
        self.sim_manager  = sim_manager
        self.data_manager = sim_manager.data_manager
        self.system       = sim_manager.system
        self.priority     = plugin_config.get('priority', 0)
        sim_manager.register_callback(
            sim_manager.pre_propagation,
            self.pre_propagation,
            self.priority,
        )
        # Load the microstate pool once at startup
        bstates_dir = os.path.join(
            os.environ.get('WEST_SIM_ROOT', '.'), 'bstates')
        self.pool_counts, self.pool_weights = _load_pool(bstates_dir)
        if self.pool_counts is not None:
            log.info('RestartDriver: loaded microstate pool with %d entries.',
                     len(self.pool_counts))
        else:
            log.info('RestartDriver: no microstate pool found; '
                     'recycled walkers will use model defaults.')

    def _sample_pool(self):
        '''Draw one row from the pool by weight. Returns 1-D float64 numpy array.'''
        if self.pool_counts is None:
            return None
        idx = np.random.choice(len(self.pool_counts), p=self.pool_weights)
        return np.array(self.pool_counts[idx], dtype=np.float64)

    def pre_propagation(self):
        segments  = self.sim_manager.incomplete_segments.values()
        n_iter    = self.sim_manager.n_iter
        if n_iter == 1:
            return
        parent_iter_group = self.data_manager.get_iter_group(n_iter - 1)
        parent_ids        = [seg.parent_id for seg in segments]
        unique_parent_ids = set(parent_ids)
        restart_data      = {sid: {} for sid in unique_parent_ids}
        try:
            dsinfo = self.data_manager.dataset_options['final_state']
        except KeyError:
            raise KeyError('Dataset final_state not found in west.h5')
        ds = parent_iter_group[dsinfo['h5path']]
        for seg_id in unique_parent_ids:
            if seg_id >= 0:
                # Normal continuing walker
                restart_data[seg_id]['restart_state']    = ds[seg_id]
                restart_data[seg_id]['pool_microstate']  = None
            else:
                # Recycled walker: sample a microstate from the pool
                sampled = self._sample_pool()
                restart_data[seg_id]['restart_state']    = [-1] * ds[0].shape[0]
                restart_data[seg_id]['pool_microstate']  = sampled
                log.debug('Recycled seg parent_id=%d: sampled pool microstate.',
                          seg_id)
        for segment in segments:
            d = restart_data[segment.parent_id]
            segment.data['restart_state']   = d['restart_state']
            segment.data['pool_microstate'] = d['pool_microstate']