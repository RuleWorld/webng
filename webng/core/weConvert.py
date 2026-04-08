from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import yaml, os, shutil, sys, westpa, bionetgen, platform, stat, warnings
import numpy as np
import nbformat as nbf

# TODO: Expose more functionality to the options file
# especially some of them can be optionally exposed
class weConvert:
    """
    This is the class that will be used by the command line tool when it's
    called with the subcommand `webng setup`.

    The class needs the dictionary from configuration YAML file for initalization
    and will use the options there to setup the WESTPA simulation folder.

    The `run` method will use the parsed options and make the WESTPA simulation folder
    using the templates it contains. TODO: Use jinja for templating instead.
    """

    def __init__(self, args):
        """
        take arguments from cement app and get ready to write
        """
        self.opts = self._load_yaml(args.opts)
        self._parse_opts(self.opts)
        # TODO: make this optional somewhere else
        self.copy_run_net = True

    def _getd(self, dic, key, default=None, required=True):
        val = dic.get(key, default)
        if required and (val is None):
            sys.exit("{} is not specified in the dictionary".format(key))
        return val

    def _parse_opts(self, opts_dict):
        """
        Parses the loaded YAML dictionary and updates the
        class attributes appropriately
        """
        # Set the main directory we are in
        self.main_dir = os.getcwd()
        # Propagator options
        propagator_options = self._getd(self.opts, "propagator_options")
        self.propagator_type = self._getd(
            propagator_options, "propagator_type", default="executable"
        )
        if self.propagator_type == "libRoadRunner":
            self.pcoord_list = self._getd(propagator_options, "pcoords")

        # we need to find WESTPA and BNG
        self.system = platform.system()
        path_options = self._getd(self.opts, "path_options")
        self.WESTPA_path = self._get_westpa_path()
        self.bng_path = self._get_bng_path()
        self.bngl_file = self._getd(path_options, "bngl_file")
        self.fname = self._getd(path_options, "sim_name", default="WE_BNG_sim")
        # Define where the BNG2.pl script is
        self.bngpl = os.path.join(self.bng_path, "BNG2.pl")

        # sampling options
        sampling_options = self._getd(self.opts, "sampling_options")
        self.tau = self._getd(sampling_options, "tau")
        self.max_iter = self._getd(sampling_options, "max_iter", default=100)
        self.dims = self._getd(sampling_options, "dimensions")
        self.plen = self._getd(sampling_options, "pcoord_length")

        # binning options
        binning_options = self._getd(self.opts, "binning_options")
        self.binning_style = self._getd(binning_options, "style", default='adaptive')
        if self.binning_style == 'adaptive':
            self.traj_per_bin = self._getd(binning_options, "traj_per_bin", default=10)
            self.block_size = self._getd(binning_options, "block_size", default=10)
            self.center_freq = self._getd(binning_options, "center_freq", default=1)
            self.max_centers = self._getd(binning_options, "max_centers", default=300)
        elif self.binning_style == "mabl":
            self.n_walkers = self._getd(binning_options, "n_walkers", default=40)
            self.n_split   = self._getd(binning_options, "n_split",   default=5)
            self.n_merge   = self.n_split + 1
            self.start     = self._getd(binning_options, "start", default=[0]*self.dims)
            self.target    = self._getd(binning_options, "target", default=[50]*self.dims)
            self.block_size = self._getd(binning_options, "block_size", default=10)
            self.coord_weights = self._getd(binning_options, "coord_weights", default=[1]*self.dims)
        else:
            self.boundaries   = self._getd(binning_options, "boundaries")
            self.traj_per_bin = self._getd(binning_options, "traj_per_bin", default=10)
            self.block_size   = self._getd(binning_options, "block_size",   default=10)

        # recycling options
        recycling_opts = opts_dict.get("recycling", None)
        self.do_recycling = recycling_opts is not None
        if self.do_recycling:
            if self.binning_style == "adaptive":
                warnings.warn(
                    "[recycling] Recycling is not supported with adaptive Voronoi "
                    "binning because bin centers shift each iteration. The recycling "
                    "block will be ignored. Switch to 'regular' or 'mabl' binning "
                    "to use recycling."
                )
                self.do_recycling = False
        if self.do_recycling:
            raw_basis = recycling_opts.get("basis_region", []) or []
            self.basis_region = []
            for c in raw_basis:
                cond = {"observable": c["observable"]}
                if "min" in c and c["min"] is not None:
                    cond["min"] = float(c["min"])
                if "max" in c and c["max"] is not None:
                    cond["max"] = float(c["max"])
                self.basis_region.append(cond)
            raw_recycle = recycling_opts.get("recycle_region", None)
            if raw_recycle is None:
                sys.exit(
                    "[recycling] recycle_region must be specified in the recycling block."
                )
            self.recycle_region = self._parse_recycle_region(raw_recycle, recycling_opts.get("_pcoord_list_hint", None))
            self.bs_traj_length = int(recycling_opts.get("traj_length",max(5000, 500 * len(self.basis_region))))
            # Validate MABL options
            if self.binning_style == "mabl":
                self._validate_mabl_config()
        else:
            self.basis_region   = []
            self.recycle_region = None
            self.bs_traj_length = 5000
    
    # Helper functions for parsing options and getting paths
    def _get_westpa_path(self):
        # full path to library
        wlib_path = westpa.__path__[0]
        # remove the last two folders, "wpath"/src/westpa
        # is the standard form of this
        wpath = os.path.split(wlib_path)[0]
        wpath = os.path.split(wpath)[0]
        return wpath

    def _get_bng_path(self):
        # now we need the BNG path, get it from the library as well
        # we need the platform and the appropriate folder name
        if self.system == "Linux":
            bng_name = "bng-linux"
        elif self.system == "Windows":
            bng_name = "bng-win"
        elif self.system == "Darwin":
            bng_name = "bng-mac"
        # get library path
        lib_path = os.path.dirname(bionetgen.__file__)
        bng_path = os.path.join(lib_path, bng_name)
        return bng_path

    def _load_yaml(self, yfile):
        """
        internal function that opens a file and loads it in using
        yaml library
        """
        with open(yfile, "r") as f:
            y = yaml.load(f, Loader=Loader)
        return y

    def _validate_mabl_config(self):
        """
        Called from _parse_opts when binning_style == 'mabl'.
        Errors out early (before any files are written) if required
        MABL options are missing.
        """
        if self.recycle_region is None:
            sys.exit(
                "ERROR: MABL binning requires 'recycle_region' to be set in "
                "binning_options.\n"
                "Without a recycling region there is no target macrostate, so "
                "no flux or rate constant can be measured.\n"
                "Example for a 2D system where dim0 <= 10 AND dim1 >= 55:\n"
                "  recycle_region: [[null, 10], [55, null]]"
            )
        if self.start is None or self.target is None:
            sys.exit(
                "ERROR: MABL binning requires both 'start' and 'target' to be "
                "set in binning_options."
            )

    def _parse_recycle_region(self, raw, pcoord_list_hint=None):
        """
        Parse recycle_region from the YAML into a per-dimension
        [min|None, max|None] list ordered to match self.pcoord_list.
        """
        pcoord_list = self.pcoord_list
        # Build result as {dim_index: [lo, hi]}, default unbounded
        result = [[None, None] for _ in pcoord_list]
        for entry in raw:
            obs = entry.get("observable")
            if obs not in pcoord_list:
                sys.exit(
                    "[recycling] recycle_region observable '{}' is not in "
                    "pcoords {}.\nMake sure observable names match exactly.".format(
                        obs, pcoord_list)
                )
            dim = pcoord_list.index(obs)
            lo  = entry.get("min", None)
            hi  = entry.get("max", None)
            result[dim] = [
                float(lo) if lo is not None else None,
                float(hi) if hi is not None else None,
            ]
        return result

    # Helper functions for parsing .net files
    def _parse_net_species(self, net_path):
        """Return list of (1-based-index, name, count) from begin species block."""
        species, in_block = [], False
        with open(net_path) as f:
            for line in f:
                s = line.strip()
                if s.startswith("begin species"):
                    in_block = True; continue
                if s.startswith("end species"):
                    break
                if in_block and s and not s.startswith("#"):
                    parts = s.split()
                    if len(parts) >= 3:
                        try:
                            species.append((int(parts[0]), parts[1], float(parts[2])))
                        except ValueError:
                            pass
        return species

    def _parse_net_groups(self, net_path):
        """
        Parse begin groups block.
        Returns dict: observable_name -> list of 1-based species indices.
        """
        groups, in_block = {}, False
        with open(net_path) as f:
            for line in f:
                s = line.strip()
                if s.startswith("begin groups"):
                    in_block = True; continue
                if s.startswith("end groups"):
                    break
                if in_block and s and not s.startswith("#"):
                    parts = s.split()
                    if len(parts) >= 3:
                        obs_name   = parts[1]
                        sp_indices = []
                        for token in parts[2].split(","):
                            token = token.strip()
                            if "*" in token:
                                token = token.split("*")[1]
                            try:
                                sp_indices.append(int(token))
                            except ValueError:
                                pass
                        if sp_indices:
                            groups[obs_name] = sp_indices
        return groups

    def _write_net_with_species(self, template_net, out_path, new_counts):
        """
        Copy template_net to out_path, replacing species counts with new_counts.
        new_counts is a list of floats in the same order as _parse_net_species.
        """
        count_iter, in_block, out_lines = iter(new_counts), False, []
        with open(template_net) as f:
            for line in f:
                s = line.strip()
                if s.startswith("begin species"):
                    in_block = True;  out_lines.append(line); continue
                if s.startswith("end species"):
                    in_block = False; out_lines.append(line); continue
                if in_block and s and not s.startswith("#"):
                    parts = s.split()
                    if len(parts) >= 3:
                        try:
                            val = next(count_iter)
                            fmt = ("{:.0f}" if val == int(val) else "{:.6g}").format(val)
                            out_lines.append(
                                "{:<6} {:<36} {}\n".format(parts[0], parts[1], fmt))
                            continue
                        except StopIteration:
                            pass
                out_lines.append(line)
        with open(out_path, "w") as f:
            f.writelines(out_lines)

    def sample_basis_states(self):
        """
        Run a long SSA via libRoadRunner, filter frames that satisfy
        basis_region conditions (evaluated on pcoords using the groups
        mapping from init.net), and write:

          bstates/microstate_pool.dat  — one line per valid frame: <weight> <sp1> <sp2> ... <spN>
          bstates/0.net                — placeholder required by w_init
          bstates/bstates.txt          — single entry: 0 1 0.net

        If recycling is disabled, copies init.net → 0.net as before and
        writes the single-entry bstates.txt.
        """
        bstates_dir = "bstates"
        init_net    = "bngl_conf/init.net"
        init_xml    = "bngl_conf/init.xml"

        # no recycle: single basis state
        if not self.do_recycling:
            shutil.copyfile(init_net, os.path.join(bstates_dir, "0.net"))
            self._write_bstatestxt(n=1)
            return

        # validate pcoords against net files
        groups      = self._parse_net_groups(init_net)
        ref_species = self._parse_net_species(init_net)
        n_species   = len(ref_species)
        for cond in self.basis_region:
            obs = cond["observable"]
            if obs not in groups:
                sys.exit(
                    "[recycling] basis_region observable '{}' not found in "
                    "groups block of {}.\nAvailable: {}".format(
                        obs, init_net, list(groups.keys()))
                )

        # run SSA using libRoadRunner
        try:
            import roadrunner as librr
        except ImportError:
            sys.exit("[recycling] libRoadRunner is required for basis-state "
                     "sampling but is not installed.")
        print("[recycling] Running sampling trajectory ({} steps) …".format(
            self.bs_traj_length))
        rr = librr.RoadRunner(init_xml)
        rr.setIntegrator("gillespie")
        rr.setIntegratorSetting("gillespie", "variable_step_size", False)
        rr.setIntegratorSetting("gillespie", "nonnegative", True)
        fs_names = rr.getFloatingSpeciesAmountsNamedArray().colnames
        rr.timeCourseSelections = (["time"]
                                   + list(self.pcoord_list)
                                   + list(fs_names))
        raw = rr.simulate(0, self.bs_traj_length, self.bs_traj_length + 1)
        n_pcoords   = len(self.pcoord_list)
        n_frames    = raw.shape[0]

        # filter frames using basis_region
        pcoord_col = {name: 1 + i for i, name in enumerate(self.pcoord_list)}
        mask = np.ones(n_frames, dtype=bool)
        for cond in self.basis_region:
            col = raw[:, pcoord_col[cond["observable"]]]
            if "min" in cond:
                mask &= col >= cond["min"]
            if "max" in cond:
                mask &= col <= cond["max"]
        valid_idx = np.where(mask)[0]
        n_valid   = len(valid_idx)
        if n_valid == 0:
            warnings.warn(
                "[recycling] No trajectory frames satisfied basis_region "
                "conditions.\n  Conditions: {}\n"
                "  Falling back to single basis state from init.net.\n"
                "  Try increasing traj_length or relaxing basis_region.".format(
                    self.basis_region)
            )
            shutil.copyfile(init_net, os.path.join(bstates_dir, "0.net"))
            self._write_bstatestxt(n=1)
            return
        print("[recycling] Found {} valid frames for basis_region.".format(n_valid))
        # write microstate_pool.dat: species columns start after time + pcoords
        sp_col_start = 1 + n_pcoords
        pool_path    = os.path.join(bstates_dir, "microstate_pool.dat")
        # Weight = 1.0 for all frames (uniform over valid frames).
        # TODO Future enhancement: weight by dwell time for equilibrium sampling.
        with open(pool_path, "w") as pf:
            pf.write("# weight " +
                     " ".join("sp{}".format(i+1) for i in range(n_species)) + "\n")
            for idx in valid_idx:
                counts = raw[idx, sp_col_start: sp_col_start + n_species]
                count_strs = " ".join(
                    ("{:.0f}" if c == int(c) else "{:.6g}").format(c) for c in counts
                )
                pf.write("1.0 {}\n".format(count_strs))
        print("[recycling] Wrote {} microstates to {}.".format(n_valid, pool_path))
        # Print observable ranges for the valid frames (useful diagnostic)
        for cond in self.basis_region:
            col  = raw[valid_idx, pcoord_col[cond["observable"]]]
            print("[recycling]   {}: min={:.1f}  max={:.1f}  mean={:.1f}".format(
                cond["observable"], col.min(), col.max(), col.mean()))
        # paceholder 0.net and bstates.txt (single entry for w_init)
        shutil.copyfile(init_net, os.path.join(bstates_dir, "0.net"))
        self._write_bstatestxt(n=1)

    def _write_tstates(self):
        """
        Write tstates.txt.

        Regular binning
        ───────────────
        Iterate over every bin defined by self.boundaries.  A bin makes contact
        with recycle_region if, for every dimension, the bin's interval overlaps
        the recycle_region interval for that dimension.  For each such bin write
        one line using the bin center as the target coordinate.  WESTPA treats
        any walker whose pcoord falls in the same bin as a target coordinate as
        recycled — so writing one point per sink bin is sufficient.

        MABL
        ────
        Write a single line with self.target as before.  The actual recycling
        trigger is the region check in MABL_driver, not this coordinate; this
        file just registers a target_state_id in the HDF5 file so w_fluxanl
        can record flux.
        """
        if not self.do_recycling:
            return

        if self.binning_style == "mabl":
            coords_str = " ".join(str(v) for v in self.target)
            with open("tstates.txt", "w") as f:
                f.write("target {}\n".format(coords_str))
            return

        # regular binning: find all bins that touch the region
        boundaries  = self.boundaries
        n_dims      = len(boundaries)
        recycle     = self.recycle_region

        if len(recycle) != n_dims:
            sys.exit(
                "[recycling] recycle_region has {} dimensions but boundaries "
                "has {}. They must match.".format(len(recycle), n_dims)
            )

        # build per-dimension list of (bin_lo, bin_hi, bin_center)
        dim_bins = []
        for d in range(n_dims):
            edges   = boundaries[d]
            # replace 'inf'/'-inf' strings with float equivalents
            edges_f = []
            for e in edges:
                if isinstance(e, str) and e.lower() in ("inf", "+inf"):
                    edges_f.append(float("inf"))
                elif isinstance(e, str) and e.lower() == "-inf":
                    edges_f.append(float("-inf"))
                else:
                    edges_f.append(float(e))
            bins_d = []
            finite_edges = [e for e in edges_f
                            if e != float("inf") and e != float("-inf")]
            if len(finite_edges) >= 2:
                typical_hw = (finite_edges[-1] - finite_edges[0]) / (len(finite_edges) - 1) / 2.0
            else:
                typical_hw = 5.0  # safe default
            for i in range(len(edges_f) - 1):
                lo  = edges_f[i]
                hi  = edges_f[i + 1]
                # center: use midpoint; cap infinities at ±1e6 for display
                lo_c = lo if lo != float("-inf") else hi - typical_hw
                hi_c = hi if hi != float("inf")  else lo + typical_hw
                bins_d.append((lo, hi, (lo_c + hi_c) / 2.0))
            dim_bins.append(bins_d)

        def bins_touch(bin_intervals):
            """
            Return True if the Cartesian bin defined by bin_intervals
            (list of (lo, hi) per dim) overlaps recycle_region.
            Overlap in dim d: bin_lo < recycle_max AND bin_hi > recycle_min
            (i.e. intervals share at least one point).
            """
            for d, (b_lo, b_hi) in enumerate(bin_intervals):
                r_lo, r_hi = recycle[d]
                # Convert None to ±inf
                r_lo = float("-inf") if r_lo is None else float(r_lo)
                r_hi = float("inf")  if r_hi is None else float(r_hi)
                if b_hi <= r_lo or b_lo >= r_hi:
                    return False
            return True

        # iterate over Cartesian product of all bins
        import itertools
        sink_centers = []
        for combo in itertools.product(*dim_bins):
            intervals = [(lo, hi) for lo, hi, _ in combo]
            if bins_touch(intervals):
                center = [ctr for _, _, ctr in combo]
                sink_centers.append(center)

        if not sink_centers:
            warnings.warn(
                "[recycling] recycle_region does not overlap any bin in "
                "boundaries. No tstates.txt entries will be written. "
                "Check that recycle_region matches your bin edges."
            )
            return

        print("[recycling] Writing {} sink bin(s) to tstates.txt.".format(
            len(sink_centers)))

        with open("tstates.txt", "w") as f:
            for i, center in enumerate(sink_centers):
                coords_str = " ".join("{:.6g}".format(v) for v in center)
                f.write("t{} {}\n".format(i, coords_str))

    def _write_librrPropagator(self):
        lines = [
            "from __future__ import division, print_function; __metaclass__ = type",
            "import numpy as np",
            "import westpa, copy, time, random, os",
            "from westpa.core.propagators import WESTPropagator",
            "from westpa.core.segment import Segment",
            "import roadrunner as librr",
            "import logging",
            "log = logging.getLogger(__name__)",
            "log.debug('loading module %r' % __name__)",
            "",
            "",
            "class librrPropagator(WESTPropagator):",
            "    def __init__(self, rc=None):",
            "        super(librrPropagator, self).__init__(rc)",
            "        config = self.rc.config",
            "        for key in [('west','librr','init','model_file'),",
            "                    ('west','librr','init','init_time_step'),",
            "                    ('west','librr','init','final_time_step'),",
            "                    ('west','librr','init','num_time_step'),",
            "                    ('west','librr','data','pcoords')]:",
            "            config.require(key)",
            "        self.runner_config = {}",
            "        self.runner_config['model_file']  = os.path.normpath(",
            "            config['west','librr','init','model_file'])",
            "        self.runner_config['init_ts']     = config['west','librr','init','init_time_step']",
            "        self.runner_config['final_ts']    = config['west','librr','init','final_time_step']",
            "        self.runner_config['num_ts']      = config['west','librr','init','num_time_step']",
            "        self.runner_config['pcoord_keys'] = config['west','librr','data','pcoords']",
            "        self.runner = librr.RoadRunner(self.runner_config['model_file'])",
            "        self.runner.setIntegrator('gillespie')",
            "        self.runner.setIntegratorSetting('gillespie', 'variable_step_size', False)",
            "        self.runner.setIntegratorSetting('gillespie', 'nonnegative', True)",
            "        self.initial_pcoord  = self.get_initial_pcoords()",
            "        self.full_state_keys = self.get_full_state_keys()",
            "        self.runner.timeCourseSelections = self.runner_config['pcoord_keys']",
            "",
            "    def get_pcoord(self, state):",
            "        state.pcoord = copy.copy(self.initial_pcoord)",
            "",
            "    def gen_istate(self, basis_state, initial_state):",
            "        # Sample a microstate from the pool and set both the",
            "        # runner state and the recorded pcoord so WESTPA's bin",
            "        # assignment reflects where the walker actually starts.",
            "        pool_path = os.path.join(",
            "            os.environ.get('WEST_SIM_ROOT', '.'), 'bstates',",
            "            'microstate_pool.dat')",
            "        if os.path.isfile(pool_path):",
            "            data = np.loadtxt(pool_path, comments='#')",
            "            if data.ndim == 1:",
            "                data = data.reshape(1, -1)",
            "            weights = data[:, 0]",
            "            counts  = data[:, 1:]",
            "            weights = weights / weights.sum()",
            "            idx     = np.random.choice(len(counts), p=weights)",
            "            pool_ms = np.array(counts[idx], dtype=np.float64)",
            "            vec     = self._build_full_state(pool_ms)",
            "            if vec is not None:",
            "                self.runner.resetAll()",
            "                self.set_runner_state(vec)",
            "        initial_state.pcoord = self.get_initial_pcoords()",
            "        return initial_state",
            "",
            "    def get_initial_pcoords(self):",
            "        return [self.runner[x] for x in self.runner_config['pcoord_keys']]",
            "",
            "    def get_full_state_keys(self):",
            "        fs    = self.runner.getFloatingSpeciesAmountsNamedArray().colnames",
            "        concs = ['[' + x + ']' for x in fs]",
            "        return fs + concs",
            "",
            "    def get_final_state(self):",
            "        return [self.runner[x] for x in self.full_state_keys]",
            "",
            "    def set_runner_state(self, state):",
            "        if all(x == -1 for x in state):",
            "            self.runner.resetAll()",
            "        else:",
            "            for i, val in enumerate(state):",
            "                self.runner.setValue(self.full_state_keys[i], val)",
            "",
            "    def _build_full_state(self, pool_microstate):",
            "        '''",
            "        Convert a pool_microstate vector (raw species amounts, length",
            "        n_species) into the full_state_keys-compatible vector",
            "        (amounts + concentrations, length 2*n_species).",
            "        For SSA models volume=1 so amount == concentration.",
            "        Returns None if length does not match.",
            "        '''",
            "        n_sp = len(self.full_state_keys) // 2",
            "        counts = list(pool_microstate)",
            "        if len(counts) != n_sp:",
            "            log.warning(",
            "                'pool_microstate length %d != expected %d; using resetAll().',",
            "                len(counts), n_sp)",
            "            return None",
            "        return counts + counts   # amounts then concentrations",
            "",
            "    def propagate(self, segments):",
            "        for segment in segments:",
            "            piter     = segment.n_iter - 1",
            "            starttime = time.time()",
            "            seed      = random.randint(0, 2**14)",
            "            self.runner.resetAll()",
            "            self.runner.setIntegratorSetting('gillespie', 'seed', seed)",
            "",
            "            if piter == 0:",
            "                # ── First iteration ─────────────────────────────",
            "                # RestartDriver does not run at iter 1, so we sample",
            "                # directly from pool_microstate if it was set during",
            "                # w_init (future), otherwise use model defaults.",
            "                pool_ms = segment.data.get('pool_microstate', None)",
            "                if pool_ms is not None:",
            "                    vec = self._build_full_state(pool_ms)",
            "                    if vec is not None:",
            "                        self.set_runner_state(vec)",
            "                        log.debug('iter1 seg %d: set state from pool.',",
            "                                  segment.seg_id)",
            "                # else: resetAll() already called — model defaults",
            "",
            "            else:",
            "                restart = segment.data.get('restart_state')",
            "                if restart is not None and not all(x == -1 for x in restart):",
            "                    # ── Normal continuing walker ─────────────────",
            "                    self.set_runner_state(restart)",
            "                else:",
            "                    # ── Recycled walker ──────────────────────────",
            "                    pool_ms = segment.data.get('pool_microstate', None)",
            "                    if pool_ms is not None:",
            "                        vec = self._build_full_state(pool_ms)",
            "                        if vec is not None:",
            "                            self.set_runner_state(vec)",
            "                            log.debug(",
            "                                'recycled seg %d: set state from pool.',",
            "                                segment.seg_id)",
            "                    # else: resetAll() already called — graceful fallback",
            "",
            "            result = self.runner.simulate(",
            "                self.runner_config['init_ts'],",
            "                self.runner_config['final_ts'],",
            "                self.runner_config['num_ts'],",
            "            )",
            "            segment.data['final_state'] = self.get_final_state()",
            "            segment.data['seed']        = seed",
            "            segment.pcoord              = result",
            "            segment.data.pop('pool_microstate', None)",
            "            segment.walltime            = time.time() - starttime",
            "            segment.cputime             = 0",
            "            segment.status              = segment.SEG_STATUS_COMPLETE",
            "        return segments",
        ]
        full_text = "\n".join(lines)
        with open("libRR_propagator.py", "w") as f:
            f.write(full_text)

    def _write_restartDriver(self):
        lines = [
            "from __future__ import division; __metaclass__ = type",
            "import logging, os, random",
            "import numpy as np",
            "log = logging.getLogger(__name__)",
            "",
            "",
            "def _load_pool(bstates_dir):",
            "    '''",
            "    Load microstate_pool.dat into a numpy array of shape (N, n_species).",
            "    Also returns the weight array of shape (N,) normalised to sum=1.",
            "    Returns (None, None) if the file does not exist (no recycling).",
            "    '''",
            "    pool_path = os.path.join(bstates_dir, 'microstate_pool.dat')",
            "    if not os.path.isfile(pool_path):",
            "        return None, None",
            "    data = np.loadtxt(pool_path, comments='#')",
            "    if data.ndim == 1:",
            "        data = data.reshape(1, -1)",
            "    weights  = data[:, 0]",
            "    counts   = data[:, 1:]",
            "    weights  = weights / weights.sum()",
            "    return counts, weights",
            "",
            "",
            "class RestartDriver(object):",
            "    def __init__(self, sim_manager, plugin_config):",
            "        super(RestartDriver, self).__init__()",
            "        if not sim_manager.work_manager.is_master:",
            "            return",
            "        self.sim_manager  = sim_manager",
            "        self.data_manager = sim_manager.data_manager",
            "        self.system       = sim_manager.system",
            "        self.priority     = plugin_config.get('priority', 0)",
            "        sim_manager.register_callback(",
            "            sim_manager.pre_propagation,",
            "            self.pre_propagation,",
            "            self.priority,",
            "        )",
            "        # Load the microstate pool once at startup",
            "        bstates_dir = os.path.join(",
            "            os.environ.get('WEST_SIM_ROOT', '.'), 'bstates')",
            "        self.pool_counts, self.pool_weights = _load_pool(bstates_dir)",
            "        if self.pool_counts is not None:",
            "            log.info('RestartDriver: loaded microstate pool with %d entries.',",
            "                     len(self.pool_counts))",
            "        else:",
            "            log.info('RestartDriver: no microstate pool found; '",
            "                     'recycled walkers will use model defaults.')",
            "",
            "    def _sample_pool(self):",
            "        '''Draw one row from the pool by weight. Returns 1-D float64 numpy array.'''",
            "        if self.pool_counts is None:",
            "            return None",
            "        idx = np.random.choice(len(self.pool_counts), p=self.pool_weights)",
            "        return np.array(self.pool_counts[idx], dtype=np.float64)",
            "",
            "    def pre_propagation(self):",
            "        segments  = self.sim_manager.incomplete_segments.values()",
            "        n_iter    = self.sim_manager.n_iter",
            "        if n_iter == 1:",
            "            return",
            "        parent_iter_group = self.data_manager.get_iter_group(n_iter - 1)",
            "        parent_ids        = [seg.parent_id for seg in segments]",
            "        unique_parent_ids = set(parent_ids)",
            "        restart_data      = {sid: {} for sid in unique_parent_ids}",
            "        try:",
            "            dsinfo = self.data_manager.dataset_options['final_state']",
            "        except KeyError:",
            "            raise KeyError('Dataset final_state not found in west.h5')",
            "        ds = parent_iter_group[dsinfo['h5path']]",
            "        for seg_id in unique_parent_ids:",
            "            if seg_id >= 0:",
            "                # Normal continuing walker",
            "                restart_data[seg_id]['restart_state']    = ds[seg_id]",
            "                restart_data[seg_id]['pool_microstate']  = None",
            "            else:",
            "                # Recycled walker: sample a microstate from the pool",
            "                sampled = self._sample_pool()",
            "                restart_data[seg_id]['restart_state']    = [-1] * ds[0].shape[0]",
            "                restart_data[seg_id]['pool_microstate']  = sampled",
            "                log.debug('Recycled seg parent_id=%d: sampled pool microstate.',",
            "                          seg_id)",
            "        for segment in segments:",
            "            d = restart_data[segment.parent_id]",
            "            segment.data['restart_state']   = d['restart_state']",
            "            segment.data['pool_microstate'] = d['pool_microstate']",
        ]
        full_text = "\n".join(lines)
        with open("restart_plugin.py", "w") as f:
            f.write(full_text)


    def _write_mabl_driver(self):
        """
        Write MABL_driver.py into the simulation folder.

        Templated constants (from your webng config):
        MABL_START          : start microstate per dimension
        MABL_TARGET         : target microstate per dimension (used for scoring)
        MABL_COORD_WEIGHTS  : per-dimension importance weights
        MABL_N_SPLIT        : walkers split per iteration
        MABL_N_MERGE        : always MABL_N_SPLIT + 1
        MABL_RECYCLE_REGION : [min, max] per dimension box, or None to disable
        """
        lines = [
            "import logging",
            "import operator",
            "import pandas as pd",
            "import numpy as np",
            "from westpa.core.we_driver import WEDriver, NewWeightEntry",
            "from westpa.core.segment import Segment",
            "",
            "log = logging.getLogger(__name__)",
            "",
            "# ---------------------------------------------------------------------------",
            "# MABL configuration — set by webng from your binning_options config.",
            "# ---------------------------------------------------------------------------",
            "",
            "MABL_START  = {}".format(self.start),
            "MABL_TARGET = {}".format(self.target),
            "",
            "# Per-dimension importance weights (relative, no normalization required).",
            "MABL_COORD_WEIGHTS = {}".format(self.coord_weights),
            "",
            "# n_merge = n_split + 1 keeps total walker count constant each iteration.",
            "MABL_N_SPLIT = {}".format(self.n_split),
            "MABL_N_MERGE = {}  # automatically MABL_N_SPLIT + 1".format(self.n_merge),
            "",
            "# Recycling region: [min, max] per dimension. None disables recycling.",
            "MABL_RECYCLE_REGION = {}".format(self.recycle_region),
            "",
            "# ---------------------------------------------------------------------------",
            "",
            "",
            "class MABLDriver(WEDriver):",
            "",
            "    # ------------------------------------------------------------------",
            "    # Helper: macrostate membership test",
            "    # ------------------------------------------------------------------",
            "",
            "    def _in_recycle_region(self, pcoord_frame):",
            "        if MABL_RECYCLE_REGION is None:",
            "            return False",
            "        for d, (lo, hi) in enumerate(MABL_RECYCLE_REGION):",
            "            val = pcoord_frame[d]",
            "            if lo is not None and val < lo:",
            "                return False",
            "            if hi is not None and val > hi:",
            "                return False",
            "        return True",
            "",
            "    # ------------------------------------------------------------------",
            "    # Override _recycle_walkers so WESTPA records flux correctly",
            "    # ------------------------------------------------------------------",
            "",
            "    def _recycle_walkers(self):",
            "        self.new_weights = []",
            "        if MABL_RECYCLE_REGION is None:",
            "            return",
            "        # During populate_initial() (w_init), _parent_map doesn't exist",
            "        # and there are no target states yet — nothing to recycle.",
            "        if not self.target_states or not hasattr(self, '_parent_map'):",
            "            return",
            "        target_state  = next(iter(self.target_states.values()))",
            "        istateiter    = iter(self.avail_initial_states.values())",
            "        used_istate_ids = set()",
            "        n_recycled      = 0",
            "        recycled_weight = 0.0",
            "        for bin in self.next_iter_binning:",
            "            for segment in set(bin):",
            "                if not self._in_recycle_region(segment.pcoord[0]):",
            "                    continue",
            "                parent = self._parent_map[segment.parent_id]",
            "                parent.endpoint_type = Segment.SEG_ENDPOINT_RECYCLED",
            "                try:",
            "                    initial_state = next(istateiter)",
            "                except StopIteration:",
            "                    raise RuntimeError(",
            "                        'MABL recycling: ran out of available initial states. '",
            "                        'Increase --segs-per-state in init.sh or reduce walkers.'",
            "                    )",
            "                istate_assignment = self.bin_mapper.assign([initial_state.pcoord])[0]",
            "                segment.parent_id   = -(initial_state.state_id + 1)",
            "                segment.pcoord[0]   = initial_state.pcoord",
            "                self.new_weights.append(",
            "                    NewWeightEntry(",
            "                        source_type       = NewWeightEntry.NW_SOURCE_RECYCLED,",
            "                        weight            = parent.weight,",
            "                        prev_seg_id       = parent.seg_id,",
            "                        prev_init_pcoord  = parent.pcoord[0].copy(),",
            "                        prev_final_pcoord = parent.pcoord[-1].copy(),",
            "                        new_init_pcoord   = initial_state.pcoord.copy(),",
            "                        target_state_id   = target_state.state_id,",
            "                        initial_state_id  = initial_state.state_id,",
            "                    )",
            "                )",
            "                bin.remove(segment)",
            "                self.next_iter_binning[istate_assignment].add(segment)",
            "                initial_state.iter_used = segment.n_iter",
            "                used_istate_ids.add(initial_state.state_id)",
            "                n_recycled      += 1",
            "                recycled_weight += parent.weight",
            "                log.debug('Recycled seg %d (w=%.4e) via istate %d',",
            "                          parent.seg_id, parent.weight, initial_state.state_id)",
            "        for state_id in used_istate_ids:",
            "            self.used_initial_states[state_id] = \\",
            "                self.avail_initial_states.pop(state_id)",
            "        log.info('Recycled %d walkers, total weight = %.4e',",
            "                 n_recycled, recycled_weight)",
            "",
            "    # ------------------------------------------------------------------",
            "    # n_istates_needed: tell WESTPA exactly how many initial states to",
            "    # pre-generate before each iteration.",
            "    # ------------------------------------------------------------------",
            "",
            "    @property",
            "    def n_istates_needed(self):",
            "        if self.final_binning is None or MABL_RECYCLE_REGION is None:",
            "            return 0",
            "        n_will_recycle = sum(",
            "            1 for b in self.final_binning",
            "            for seg in b",
            "            if self._in_recycle_region(seg.pcoord[-1])",
            "        )",
            "        n_avail = len(self.avail_initial_states)",
            "        return max(0, n_will_recycle - n_avail)",
            "",
            "    # ------------------------------------------------------------------",
            "    # Split / merge helpers",
            "    # ------------------------------------------------------------------",
            "",
            "    def _split_by_data(self, bin, to_split, split_into):",
            "        if len(to_split) > 1:",
            "            for segment in to_split:",
            "                bin.remove(segment)",
            "                new_segments_list = self._split_walker(segment, split_into, bin)",
            "                bin.update(new_segments_list)",
            "        else:",
            "            to_split = to_split[0]",
            "            bin.remove(to_split)",
            "            new_segments_list = self._split_walker(to_split, split_into, bin)",
            "            bin.update(new_segments_list)",
            "",
            "    def _merge_by_data(self, bin, to_merge):",
            "        bin.difference_update(to_merge)",
            "        new_segment, parent = self._merge_walkers(to_merge, None, bin)",
            "        bin.add(new_segment)",
            "",
            "    # ------------------------------------------------------------------",
            "    # Main WE loop",
            "    # ------------------------------------------------------------------",
            "",
            "    def _run_we(self):",
            "        self.new_weights = self.new_weights or []",
            "        self._recycle_walkers()",
            "        self._check_pre()",
            "",
            "        for bin in self.next_iter_binning:",
            "            if len(bin) == 0:",
            "                continue",
            "",
            "            segments = np.array(",
            "                sorted(bin, key=operator.attrgetter('weight')),",
            "                dtype=np.object_",
            "            )",
            "            weights     = np.array(list(map(operator.attrgetter('weight'), segments)))",
            "            log_weights = -1.0 * np.log(weights)",
            "",
            "            all_pcoords = np.array(",
            "                list(map(operator.attrgetter('pcoord'), segments))",
            "            )",
            "            nsegs, nframes, ndims = all_pcoords.shape",
            "",
            "            # --- Progress scores ---",
            "            progresses = np.zeros((nsegs, ndims), dtype=float)",
            "            for d in range(ndims):",
            "                span = abs(MABL_START[d] - MABL_TARGET[d])",
            "                if span == 0:",
            "                    progresses[:, d] = 1.0",
            "                    continue",
            "                for i in range(nsegs):",
            "                    dist = abs(all_pcoords[i, 0, d] - MABL_TARGET[d]) / span",
            "                    progresses[i, d] = max(0.0, 1.0 - dist)",
            "",
            "            weighted_prog = progresses * np.array(MABL_COORD_WEIGHTS)",
            "            combined      = np.prod(weighted_prog, axis=1) * (1.0 / log_weights)",
            "",
            "            # --- Skip resampling on initialisation iteration ---",
            "            curr_segs = np.array(",
            "                sorted(self.current_iter_segments, key=operator.attrgetter('weight')),",
            "                dtype=np.object_",
            "            )",
            "            curr_pc0 = np.array(",
            "                list(map(operator.attrgetter('pcoord'), curr_segs))",
            "            )[:, :, 0]",
            "            if not np.any(curr_pc0[:, 0] != curr_pc0[:, -1]):",
            "                continue",
            "",
            "            # --- Split top MABL_N_SPLIT in-bounds walkers ---",
            "            to_split_idx = []",
            "            for idx in np.argsort(-combined):",
            "                if combined[idx] <= 0:",
            "                    continue",
            "                if segments[idx].parent_id < 0:",
            "                    continue",
            "                in_bounds = all(",
            "                    all_pcoords[idx, 0, d] < MABL_TARGET[d]",
            "                    if MABL_TARGET[d] > MABL_START[d]",
            "                    else all_pcoords[idx, 0, d] > MABL_TARGET[d]",
            "                    for d in range(ndims)",
            "                )",
            "                if in_bounds:",
            "                    to_split_idx.append(idx)",
            "                if len(to_split_idx) >= MABL_N_SPLIT:",
            "                    break",
            "",
            "            if to_split_idx:",
            "                to_split = np.array([segments[to_split_idx]])[0]",
            "                self._split_by_data(bin, to_split, 2)",
            "",
            "            # Rebuild after split",
            "            segments = np.array(",
            "                sorted(bin, key=operator.attrgetter('weight')),",
            "                dtype=np.object_",
            "            )",
            "            weights     = np.array([s.weight for s in segments])",
            "            log_weights = -1.0 * np.log(weights)",
            "            all_pcoords = np.array([s.pcoord for s in segments])",
            "            nsegs       = len(segments)",
            "",
            "            # Recompute scores on fresh segments",
            "            progresses = np.zeros((nsegs, ndims), dtype=float)",
            "            for d in range(ndims):",
            "                span = abs(MABL_START[d] - MABL_TARGET[d])",
            "                if span == 0:",
            "                    progresses[:, d] = 1.0",
            "                    continue",
            "                for i in range(nsegs):",
            "                    dist = abs(all_pcoords[i, 0, d] - MABL_TARGET[d]) / span",
            "                    progresses[i, d] = max(0.0, 1.0 - dist)",
            "",
            "            weighted_prog = progresses * np.array(MABL_COORD_WEIGHTS)",
            "            combined      = np.prod(weighted_prog, axis=1) * (1.0 / log_weights)",
            "",
            "            # --- Merge bottom MABL_N_MERGE in-bounds walkers ---",
            "            to_merge_idx = []",
            "            for idx in np.argsort(combined):",
            "                if combined[idx] <= 0:",
            "                    continue",
            "                if segments[idx].parent_id < 0:",
            "                    continue",
            "                in_bounds = all(",
            "                    all_pcoords[idx, 0, d] < MABL_TARGET[d]",
            "                    if MABL_TARGET[d] > MABL_START[d]",
            "                    else all_pcoords[idx, 0, d] > MABL_TARGET[d]",
            "                    for d in range(ndims)",
            "                )",
            "                if in_bounds:",
            "                    to_merge_idx.append(idx)",
            "                if len(to_merge_idx) >= MABL_N_MERGE:",
            "                    break",
            "",
            "            if to_merge_idx:",
            "                self._merge_by_data(bin, segments[to_merge_idx])",
            "",
            "        self._check_post()",
            "        log.debug('used initial states: {!r}'.format(self.used_initial_states))",
            "        log.debug('available initial states: {!r}'.format(self.avail_initial_states))",
        ]
        full_text = "\n".join(lines)
        with open("MABL_driver.py", "w") as f:
            f.write(full_text)

    def _write_runsh(self):
        """
        write the run.sh file for WESTPA simulations
        """
        # TODO: Add submission scripts for varied clusters
        # TODO: Add a hook to write any submission scripts?
        lines = ["#!/bin/bash", 'w_run --work-manager processes "$@"']
        
        full_text = "\n".join(lines)
        with open("run.sh", "w") as f:
            f.write(full_text)
        os.chmod("run.sh", 0o764)

    # def _write_runps1(self):
    #     """
    #     write the run.sh file for WESTPA simulations - WINDOWS VERSION
    #     """
    #     lines = ["w_run --work-manager processes $args"]
        
    #     full_text = "\n".join(lines)
    #     with open("run.ps1", "w") as f:
    #         f.write(full_text)
    #     os.chmod("run.ps1", 0o764)

    def _write_envsh(self):
        """
        environment script that uses westpa.sh to setup the environment - Unix
        """
        if self.WESTPA_path is None:
            sys.exit("WESTPA path is not specified")

        lines = [
            "#!/bin/sh",
            'export WEST_SIM_ROOT="$PWD"',
            "export SIM_NAME=$(basename $WEST_SIM_ROOT)",
        ]

        if self.copy_run_net:
            lines.append('export RunNet="$WEST_SIM_ROOT/bngl_conf/run_network"')
        else:
            lines.append('export RunNet="{}/bin/run_network"'.format(self.bng_path))

        full_text = "\n".join(lines)
        with open("env.sh", "w") as f:
            f.write(full_text)
        os.chmod("env.sh", 0o764)

    # def _write_envps1(self):
    #     """
    #     environment script that uses westpa.sh to setup the environment - WINDOWS VERSION
    #     """
    #     if self.WESTPA_path is None:
    #         sys.exit("WESTPA path is not specified")

    #     lines = [
    #         "$env:WEST_SIM_ROOT = (Get-Location).Path",
    #         '$env:SIM_NAME = Split-Path -Leaf $env:WEST_SIM_ROOT',
    #     ]

    #     if self.copy_run_net:
    #         lines.append('$env:RunNet = Join-Path $env:WEST_SIM_ROOT "bngl_conf\run_network"')
    #     else:
    #         lines.append('$env:RunNet = "{}/bin/run_network"'.format(self.bng_path))

    #     full_text = "\n".join(lines)
    #     with open("env.ps1", "w") as f:
    #         f.write(full_text)
    #     os.chmod("env.ps1", 0o764)

    def _write_auxfuncs(self):
        """
        auxilliary function, by default we want to avoid the first point because that's
        time in BNG output
        """
        lines = [
            "#!/usr/bin/env python",
            "import numpy",
            "def pcoord_loader(fieldname, coord_filename, segment, single_point=False):",
            "    pcoord    = numpy.loadtxt(coord_filename, dtype = numpy.float32)",
            "    if not single_point:",
            "        segment.pcoord = pcoord[:,1:]",
            "    else:",
            "        segment.pcoord = pcoord[1:]",
        ]

        full_text = "\n".join(lines)
        with open("aux_functions.py", "w") as f:
            f.write(full_text)

    def _write_bstatestxt(self, n=1):
        lines = ["{} 1 {}.net\n".format(i, i) for i in range(n)]
        with open("bstates/bstates.txt", "w") as f:
            f.writelines(lines)

    def _write_getpcoord_sh(self):
        """
        the pcoord acquiring script for the inital center
        """
        lines = [
            "#!/bin/bash\n",
            'if [ -n "$SEG_DEBUG" ] ; then',
            "  set -x",
            "  env | sort",
            "fi",
            "cd $WEST_SIM_ROOT",
            "cat bngl_conf/init.gdat > $WEST_PCOORD_RETURN",
            'if [ -n "$SEG_DEBUG" ] ; then',
            "  head -v $WEST_PCOORD_RETURN",
            "fi",
        ]
        # OSDEPEND: EXECUTABLE - This won't work on Windows powershell

        full_text = "\n".join(lines)
        with open("westpa_scripts/get_pcoord.sh", "w") as f:
            f.write(full_text)
        os.chmod("westpa_scripts/get_pcoord.sh", 0o764)

    def _write_postiter_sh(self):
        """
        a basic post-iteration script that deletes iterations that are
        older than 3 iterations
        """
        lines = [
            "#!/bin/bash",
            'if [ -n "$SEG_DEBUG" ] ; then',
            "    set -x",
            "    env | sort",
            "fi",
            "cd $WEST_SIM_ROOT || exit 1",
            "if [[ $WEST_CURRENT_ITER -gt 3 ]];then",
            '  PREV_ITER=$(printf "%06d" $((WEST_CURRENT_ITER-3)))',
            "  rm -rf ${WEST_SIM_ROOT}/traj_segs/${PREV_ITER}",
            "  rm -f  seg_logs/${PREV_ITER}-*.log",
            "fi",
        ]
        # OSDEPEND: EXECUTABLE - This won't work on Windows powershell

        full_text = "\n".join(lines)
        with open("westpa_scripts/post_iter.sh", "w") as f:
            f.write(full_text)
        os.chmod("westpa_scripts/post_iter.sh", 0o764)

    def _write_initsh(self, traj=True):
        """
        WESTPA initialization script for Unix
        """
        segs = self.n_walkers if self.binning_style == "mabl" else self.traj_per_bin

        if self.do_recycling:
            tstate_line = 'TSTATE_ARGS="--tstate-file $WEST_SIM_ROOT/tstates.txt"'
            tstate_arg  = "$TSTATE_ARGS "
        else:
            tstate_line = ""
            tstate_arg  = ""

        base = ["#!/bin/bash", "source env.sh"]
        if traj:
            base += ["rm -rf traj_segs seg_logs istates west.h5",
                     "mkdir   seg_logs traj_segs"]
        else:
            base += ["rm -rf istates west.h5"]

        base.append('BSTATE_ARGS="--bstate-file bstates/bstates.txt"')
        if tstate_line:
            base.append(tstate_line)
        base.append(
            'w_init $BSTATE_ARGS {}--segs-per-state {} --work-manager=threads "$@"'.format(
                tstate_arg, segs)
        )

        with open("init.sh", "w") as f:
            f.write("\n".join(base))
        os.chmod("init.sh", 0o764)

    # def _write_initps1(self, traj=True):
    #     """
    #     WESTPA initialization script - WINDOWS VERSION
    #     """
    #     if traj:
    #         lines = [
    #             ". .\env.ps1",
    #             "Remove-Item -Recurse -Force traj_segs, seg_logs, istates, west.h5 -ErrorAction SilentlyContinue",
    #             "New-Item -ItemType Directory -Name seg_logs, traj_segs | Out-Null",
    #             'Copy-Item "$env:WEST_SIM_ROOT\\bngl_conf\init.net" -Destination "bstates\\0.net"',
    #             '$BSTATE_ARGS = @("--bstate-file", "bstates/bstates.txt")',
    #             "w_init @BSTATE_ARGS --segs-per-state {} --work-manager=threads $args".format(
    #                 self.traj_per_bin
    #             )
    #         ]
    #     else:
    #         lines = [
    #             ". .\env.ps1",
    #             "Remove-Item -Recurse -Force istates, west.h5 -ErrorAction SilentlyContinue",
    #             'Copy-Item "$env:WEST_SIM_ROOT\\bngl_conf\init.net" -Destination "bstates\\0.net"',
    #             '$BSTATE_ARGS = @("--bstate-file", "bstates/bstates.txt")',
    #             "w_init @BSTATE_ARGS --segs-per-state {} --work-manager=threads $args".format(
    #                 self.traj_per_bin
    #             )
    #         ]
        
    #     full_text = "\n".join(lines)
    #     with open("init.ps1", "w") as f:
    #         f.write(full_text)
    #     os.chmod("init.ps1", 0o764)

    def _write_systempy(self):
        if self.binning_style == "mabl":
            lines = [
                "from __future__ import division, print_function; __metaclass__ = type",
                "import numpy as np",
                "import westpa",
                "from westpa import WESTSystem",
                "from westpa.core.binning.assign import BinMapper",
                "import logging",
                "log = logging.getLogger(__name__)",
                "log.debug('loading module %r' % __name__)",
                "",
                "",
                "class SingleBinMapper(BinMapper):",
                "    '''",
                "    Two-bin mapper for MABL.",
                "    Bin 0: all walkers.",
                "    Bin 1: dummy target bin so report_bin_statistics does not",
                "           divide by zero when a target state is registered.",
                "    The target pcoord maps to bin 1; everything else maps to bin 0.",
                "    MABLDriver._recycle_walkers() handles all recycling logic.",
                "    '''",
                "",
                "    TARGET_PCOORD = np.array({}, dtype=float)".format(self.target),
                "",
                "    def __init__(self):",
                "        super().__init__()",
                "        self.nbins = 2",
                "",
                "    def assign(self, coords, mask=None, output=None):",
                "        coords = np.asarray(coords)",
                "        n = len(coords)",
                "        if output is None:",
                "            output = np.zeros(n, dtype=np.intp)",
                "        else:",
                "            output[:] = 0",
                "        for i in range(n):",
                "            if np.allclose(coords[i], self.TARGET_PCOORD):",
                "                output[i] = 1",
                "        return output",
                "",
                "",
                "class System(WESTSystem):",
                "    def initialize(self):",
                "        self.pcoord_ndim = {}".format(self.dims),
                "        self.pcoord_len  = {}".format(self.plen + 1),
                "        self.bin_mapper  = SingleBinMapper()",
                "        self.bin_target_counts = np.empty((self.bin_mapper.nbins,), int)",
                "        self.bin_target_counts[0] = {}   # all real walkers".format(self.n_walkers),
                "        self.bin_target_counts[1] = 1    # dummy target bin",
            ]
            full_text = "\n".join(lines)

        elif self.binning_style == "regular":
            lines = [
                "from __future__ import division, print_function; __metaclass__ = type",
                "import numpy as np",
                "import westpa",
                "from westpa import WESTSystem",
                "from westpa.core.binning import RectilinearBinMapper",
                "import logging",
                "log = logging.getLogger(__name__)",
                "log.debug('loading module %r' % __name__)",
                "class System(WESTSystem):",
                "    def initialize(self):",
                "        self.pcoord_ndim = {}".format(self.dims),
                "        self.pcoord_len  = {}".format(self.plen + 1),
                "        boundaries = {}".format(self.boundaries),
                "        self.bin_mapper = RectilinearBinMapper(boundaries)",
                "        self.bin_target_counts = np.empty((self.bin_mapper.nbins,), int)",
                "        self.bin_target_counts[...] = {}".format(self.traj_per_bin),
            ]
            full_text = "\n".join(lines)

        else:  # adaptive
            lines = [
                "from __future__ import division, print_function; __metaclass__ = type",
                "import numpy as np",
                "import westpa",
                "from westpa import WESTSystem",
                "from westpa.core.binning import VoronoiBinMapper",
                "from scipy.spatial.distance import cdist",
                "import logging",
                "log = logging.getLogger(__name__)",
                "log.debug('loading module %r' % __name__)",
                "def dfunc(p, centers):",
                "    ds = cdist(np.array([p]),centers)",
                "    return np.array(ds[0], dtype=p.dtype)",
                "class System(WESTSystem):",
                "    def initialize(self):",
                "        self.pcoord_ndim = {}".format(self.dims),
                "        self.pcoord_len  = {}".format(self.plen + 1),
                "        self.nbins = 1",
                "        centers = np.zeros((self.nbins,self.pcoord_ndim),dtype=np.float32)",
                "        i = np.loadtxt('bngl_conf/init.gdat')",
                "        centers[0] = i[0,1:]",
                "        self.bin_mapper = VoronoiBinMapper(dfunc, centers)",
                "        self.bin_target_counts = np.empty((self.bin_mapper.nbins,), int)",
                "        self.bin_target_counts[...] = {}".format(self.traj_per_bin),
            ]
            full_text = "\n".join(lines)

        with open("system.py", "w") as f:
            f.write(full_text)

    def _write_westcfg(self):
        """
        the WESTPA configuration file, another YAML file
        """
        # TODO: Expose max wallclock time?
        if self.propagator_type == "executable":
            self._executable_westcfg()
        elif self.propagator_type == "libRoadRunner":
            self._libRR_westcfg()

    def _libRR_westcfg(self):
        step_len = self.tau / self.plen
        step_no = self.plen

        if self.binning_style == "mabl":
            drivers_block  = [
                "  drivers:",
                "    module_path: $WEST_SIM_ROOT",
                "    we_driver: MABL_driver.MABLDriver",
            ]
            plugin_insert = []
        elif self.binning_style == "adaptive":
            drivers_block  = []
            plugin_insert = [
                "    - plugin: westpa.westext.adaptvoronoi.AdaptiveVoronoiDriver",
                "      av_enabled: true",
                "      dfunc_method: system.dfunc",
                "      walk_count: {}".format(self.traj_per_bin),
                "      max_centers: {}".format(self.max_centers),
                "      center_freq: {}".format(self.center_freq),
            ]
        else:
            drivers_block  = []
            plugin_insert = []

        gen_istates = "true" if self.do_recycling else "false"

        lines = (
            ["# vi: set filetype=yaml :","---","west:"]
            + drivers_block
            + [
            "  system:",
            "    driver: system.System",
            "    module_path: $WEST_SIM_ROOT",
            "  propagation:",
            "    max_total_iterations: {}".format(self.max_iter),
            "    max_run_wallclock:    72:00:00",
            "    propagator:           libRR_propagator.librrPropagator ",
            "    gen_istates:          {}".format(gen_istates),
            "    block_size:           {}".format(self.block_size),
            "  data:\n",
            "    west_data_file: west.h5",
            "    datasets:",
            "      - name:        pcoord",
            "        scaleoffset: 4",
            "      - name:        seed",
            "        scaleoffset: 4",
            "      - name:        final_state",
            "        scaleoffset: 4",
            "      - name:        pool_microstate",
            "        scaleoffset: 4",
            "  plugins:"] 
            + plugin_insert 
            + ["    - plugin: restart_plugin.RestartDriver",
            "  librr:",
            "    init:",
            "      model_file: ./bngl_conf/init.xml",
            "      init_time_step: 0",
            "      final_time_step: {}".format(self.tau),
            "      num_time_step: {}".format(step_no + 1),
            "    data:",
            "      pcoords: {}".format(('["' + '","'.join(self.pcoord_list) + '"]')),
            ]
        )  
        # TODO: Write pcoords

        full_text = "\n".join(lines)
        with open("west.cfg", "w") as f:
            f.write(full_text)

    def _executable_westcfg(self):
        if self.binning_style == "mabl":
            drivers_block  = [
                "  drivers:",
                "    module_path: $WEST_SIM_ROOT",
                "    we_driver: MABL_driver.MABLDriver",
            ]
            plugin_insert = []
        elif self.binning_style == "adaptive":
            drivers_block  = []
            plugin_insert = [
                "    - plugin: westpa.westext.adaptvoronoi.AdaptiveVoronoiDriver",
                "      av_enabled: true",
                "      dfunc_method: system.dfunc",
                "      walk_count: {}".format(self.traj_per_bin),
                "      max_centers: {}".format(self.max_centers),
                "      center_freq: {}".format(self.center_freq),
            ]
        else:
            drivers_block  = []
            plugin_insert = []

        lines = (
            ["# vi: set filetype=yaml :"]
            + drivers_block
            + [
            "---",
            "west:",
            "  system:",
            "    driver: system.System",
            "    module_path: $WEST_SIM_ROOT",
            "  propagation:",
            "    max_total_iterations: {}".format(self.max_iter),
            "    max_run_wallclock:    72:00:00",
            "    propagator:           executable",
            "    gen_istates:          false",
            "    block_size:           {}".format(self.block_size),
            "  data:",
            "    west_data_file: west.h5",
            "    datasets:",
            "      - name:        pcoord",
            "        scaleoffset: 4",
            "    data_refs:\n",
            "      segment:       $WEST_SIM_ROOT/traj_segs/{segment.n_iter:06d}/{segment.seg_id:06d}",
            "      basis_state:   $WEST_SIM_ROOT/bstates/{basis_state.auxref}",
            "      initial_state: $WEST_SIM_ROOT/istates/{initial_state.iter_created}/{initial_state.state_id}.rst",
            "  plugins:\n"] 
            + plugin_insert 
            + ["  executable:",
            "    environ:",
            "      PROPAGATION_DEBUG: 1",
            "    datasets:",
            "      - name:    pcoord",
            "        loader:  aux_functions.pcoord_loader",
            "        enabled: true",
            "    propagator:",
            "      executable: $WEST_SIM_ROOT/westpa_scripts/runseg.sh",
            "      stdout:     $WEST_SIM_ROOT/seg_logs/{segment.n_iter:06d}-{segment.seg_id:06d}.log",
            "      stderr:     stdout",
            "      stdin:      null",
            "      cwd:        null",
            "      environ:",
            "        SEG_DEBUG: 1",
            "    get_pcoord:",
            "      executable: $WEST_SIM_ROOT/westpa_scripts/get_pcoord.sh",
            "      stdout:     /dev/null ",
            "      stderr:     stdout",
            "    gen_istate:",
            "      executable: $WEST_SIM_ROOT/westpa_scripts/gen_istate.sh",
            "      stdout:     /dev/null",
            "      stderr:     stdout",
            "    post_iteration:",
            "      enabled:    true",
            "      executable: $WEST_SIM_ROOT/westpa_scripts/post_iter.sh",
            "      stderr:     stdout",
            "    pre_iteration:",
            "      enabled:    false",
            "      executable: $WEST_SIM_ROOT/westpa_scripts/pre_iter.sh",
            "      stderr:     stdout",
            ]
        )
        # OSDEPEND: EXECUTABLE - This refers to Bash scripts only Unix can use
        full_text = "\n".join(lines)
        with open("west.cfg", "w") as f:
            f.write(full_text)

    def _write_runsegsh(self):
        """
        the most important script that extends an individual segment,
        this is where tau is defined
        """

        step_len = self.tau / self.plen
        step_no = self.plen

        lines = [
            "#!/bin/bash\n",
            'if [ -n "$SEG_DEBUG" ] ; then',
            "  set -x",
            "  env | sort",
            "fi",
            "if [[ -n $SCRATCH ]];then",
            "  mkdir -pv $WEST_CURRENT_SEG_DATA_REF",
            "  mkdir -pv ${SCRATCH}/$WEST_CURRENT_SEG_DATA_REF",
            "  cd ${SCRATCH}/$WEST_CURRENT_SEG_DATA_REF",
            "else",
            "  mkdir -pv $WEST_CURRENT_SEG_DATA_REF",
            "  cd $WEST_CURRENT_SEG_DATA_REF",
            "fi",
            'if [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_CONTINUES" ]; then',
            "  if [[ -n $SCRATCH ]];then",
            "    cp $WEST_PARENT_DATA_REF/seg_end.net ./parent.net",
            "    cp $WEST_PARENT_DATA_REF/seg.gdat ./parent.gdat",
            "  else",
            "    ln -sv $WEST_PARENT_DATA_REF/seg_end.net ./parent.net",
            "    ln -sv $WEST_PARENT_DATA_REF/seg.gdat ./parent.gdat",
            "  fi",
            "  $RunNet -o ./seg -p ssa -h $WEST_RAND16 --cdat 0 --fdat 0 -x -e -g ./parent.net ./parent.net {} {}".format(
                step_len, step_no
            ),
            "  tail -n 1 parent.gdat > $WEST_PCOORD_RETURN",
            "  cat seg.gdat >> $WEST_PCOORD_RETURN",
            'elif [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_NEWTRAJ" ]; then',
            "  if [[ -n $SCRATCH ]];then",
            "    cp $WEST_PARENT_DATA_REF ./parent.net",
            "  else",
            "    ln -sv $WEST_PARENT_DATA_REF ./parent.net",
            "  fi",
            "  $RunNet -o ./seg -p ssa -h $WEST_RAND16 --cdat 0 --fdat 0 -e -g ./parent.net ./parent.net {} {}".format(
                step_len, step_no
            ),
            "  cat seg.gdat > $WEST_PCOORD_RETURN",
            "fi",
            "if [[ -n $SCRATCH ]];then",
            "  cp ${SCRATCH}/$WEST_CURRENT_SEG_DATA_REF/seg_end.net $WEST_CURRENT_SEG_DATA_REF/.",
            "  rm -rf ${SCRATCH}/$WEST_CURRENT_SEG_DATA_REF",
            "fi"
        ]
        # OSDEPEND: EXECUTABLE - This won't work on Windows powershell
        full_text = "\n".join(lines)
        with open("westpa_scripts/runseg.sh", "w") as f:
            f.write(full_text)
        os.chmod("westpa_scripts/runseg.sh", 0o764)

    def write_dynamic_files(self):
        """
        these files change depending on the given options, in particular
        sampling and binning options
        """
        self._write_systempy()
        self._write_westcfg()
        # if self.system == "Windows":
        #     if self.propagator_type == "executable":
        #         self._write_runsegps1() # OS
        #         self._write_initps1(traj=True) # OS
        #     else:
        #         self._write_initps1(traj=False) # OS
        if self.propagator_type == "executable":
            self._write_runsegsh()
            self._write_initsh(traj=True)
        else:
            self._write_initsh(traj=False)
         
    def write_static_files(self):
        """
        these files are always (mostly) the same regardless of given options
        """
        # if self.system == 'Windows':
        #     self._write_envps1() # OS
        #     self._write_bstatestxt()
        #     self._write_auxfuncs()
        #     self._write_runps1() # OS
        #     if self.propagator_type == "executable":
        #         self._write_getpcoord_ps1() # OS
        #         self._write_postiter_ps1() # OS
        #     elif self.propagator_type == "libRoadRunner":
        #         self._write_restartDriver()
        #         self._write_librrPropagator()  
        self._write_envsh()
        self._write_auxfuncs()
        self._write_runsh()
        if self.propagator_type == "executable":
            self._write_getpcoord_sh()
            self._write_postiter_sh()
        elif self.propagator_type == "libRoadRunner":
            self._write_restartDriver()
            self._write_librrPropagator()
        if self.binning_style == "mabl":
            self._write_mabl_driver()
        if self.do_recycling:
            self._write_tstates()

    def make_sim_folders(self):
        """
        make folders WESTPA needs
        """
        self.sim_dir = self.fname
        try:
            os.makedirs(self.fname)
        except FileExistsError as e:
            # TODO: make an overwrite option
            print(f"The folder {self.fname} you are trying to create already exists")
            print(e)
        os.chdir(self.fname)
        os.makedirs("bngl_conf")
        os.makedirs("bstates")
        if self.propagator_type == "executable":
            os.makedirs("westpa_scripts")

    def copy_run_network(self):
        """
        this copies the run_network binary with correct permissions to where
        WESTPA will expect to find it.
        """
        run_network = "run_network.exe" if self.system == "Windows" else "run_network"
        source = os.path.join(self.bng_path, "bin", run_network)
        destination = os.path.join("bngl_conf", run_network)
        shutil.copyfile(source, destination)
        if self.system != "Windows":
            os.chmod(destination, 0o764)
        else:
            os.chmod(destination, stat.S_IWRITE)

    def run_BNGL_on_file(self):
        """
        this function runs the BNG2.pl on the given bngl file
        to get a) .net file for the starting point and b) .gdat file
        to get the first voronoi center for the simulation
        """
        if self.propagator_type == "executable":
            self._executable_BNGL_on_file()
        elif self.propagator_type == "libRoadRunner":
            self._libRR_BNGL_on_file()

    def _libRR_BNGL_on_file(self):
        # We still need this stuff
        model = self._executable_BNGL_on_file()
        # But we also need to generate the XML file
        # get in the conf folder
        os.chdir("bngl_conf")
        # make a copy that we will use to generate the XML
        sim = model.setup_simulator()
        sbml_str = sim.getCurrentSBML()
        with open("init.xml", "w") as f:
            f.write(str(sbml_str))
        os.chdir(os.path.join(self.main_dir, self.sim_dir))

    def _executable_BNGL_on_file(self):
        # IMPORTANT!
        # This assumes that the bngl file doesn't have any directives at the end!
        # we have a bngl file
        # Make specific BNGL files for a) generating network and then
        # b) getting a starting  gdat file
        model = bionetgen.bngmodel(f"../{self.bngl_file}")
        model.add_action("generate_network", action_args={"overwrite": 1})
        model.add_action(
            "simulate", action_args={"method": "'ssa'", "t_end": 2, "n_steps": 2}
        )
        os.chdir("bngl_conf")
        with open("init.bngl", "w") as f:
            f.write(str(model))
        r = bionetgen.run("init.bngl", "for_init")
        shutil.copyfile(os.path.join("for_init", "init.net"), "init.net")
        header_str = ""
        for i in r[0].dtype.names:
            header_str += " " + i
        np.savetxt("init.gdat", r[0], header=header_str)
        shutil.rmtree("for_init")
        os.chdir(os.path.join(self.main_dir, self.sim_dir))
        return model
    
    def make_analysis_notebook(self):
        nb = nbf.v4.new_notebook()

        nb.cells.append(nbf.v4.new_markdown_cell('''# WEBNG Analysis Jupyter Notebook
Written by Alex DiBiasi'''))
        nb.cells.append(nbf.v4.new_code_cell('''import os
import matplotlib.pyplot as plt
import webng.analysis as wb
curr_path = os.getcwd()'''))
        nb.cells.append(nbf.v4.new_markdown_cell('''## Options that apply to all analysis tools'''))
        nb.cells.append(nbf.v4.new_code_cell(f'''analsyis_opts = {{
    'pcoords': ['S2_A','S2_B'],     # BNGL observables / WESTPA progress coordinates. These should not need to be changed
    'sim_name': '{self.fname}',             # WESTPA simulation folder. This notebook should be in this folder. Also does not need changed
    'analysis_bins': 30,            # Smoothness of the analysis histograms
    'first-iter': None,             # Perform analyses starting from this iteration. 'None' will default to the first iteration.
    'last-iter': None,              # Perform analyses ending at this iteration. 'None' will default to the last iteration.
    'work-path': None               # Analysis folder that is created. 'None' will default to 'analysis'
}}'''))
        nb.cells.append(nbf.v4.new_markdown_cell('''## Average Analysis
This analysis creates *pdist.h5*, which contains all of the probability distributions for each progress coordinate for each iteration. The average analysis produces probability distributions for each progress coordinate over the iterations specified above by averaging them. The resulting plot shows 1D distributions for each progress coordinate. If there is more than 1 progress coordinate, 2D probability distrbitions will be plotted using every possible pair of progress coordinates'''))
        nb.cells.append(nbf.v4.new_code_cell('''average_opts = {{
    **analsyis_opts,
    'config_file': None,            # Path to your webng YAML config file.
                                    # Required to plot bin boundaries for 'regular' binning.
                                    # Example: '../mysim.yaml'
    'mapper-iter': None,            # Specifies which iteration to take adaptive voronoi bins to plot
    'plot-boundaries': False,          # Plots bin boundaries (voronoi for adaptive, lines for regular)
    'plot-energy': False,           # If true, will plot energy by taking the negative log of the probability. E ~ -log(P)
    'normalize': False,             # Normalize the distirbutions so that they range from 0-1
    'dimensions': None,             # Specifies the first X dimensions to plot. 'None' will plot all dimensions
    'output': 'average.png',        # Name of the output file
    'smoothing': 0.5,               # Histogram smoothing option
    'color_bar': True,              # Plots a color bar for each heatmap
    'plot-opts':{{                   
        'name-font-size': 12,       # Font size
        'line_width': 1,            # Line width of bin boundaries (voronoi or regular)
        'line-col': 0.75            # Line color of bin boundaries (voronoi or regular)
    }}
}}'''))
        nb.cells.append(nbf.v4.new_code_cell('''os.chdir("..")
avg_obj = wb.average.weAverage(average_opts)
try:
    f, axarr = avg_obj.run()
except Exception as e:
    print(e)
os.chdir(curr_path)'''))
        nb.cells.append(nbf.v4.new_markdown_cell('''## Evolution Analysis
This analysis creates *pdist.h5*, which contains all of the probability distributions for each progress coordinate for each iteration. The evolution analysis produces probability distributions for each progress coordinate for each iteration in the specified iterations. The resulting plot shows 1D distributions for each progress coordinate for each iteration.'''))
        nb.cells.append(nbf.v4.new_code_cell('''evolution_opts = {
    **analsyis_opts,
    'plot-energy': False,           # If true, will plot energy by taking the negative log of the probability. E ~ -log(P)
    'normalize': False,             # Normalize the distirbutions so that they range from 0-1
    'dimensions': None,             # Specifies the first X dimensions to plot. 'None' will plot all dimensions
    'output': 'evolution.png',      # Name of the output file
    'avg_window': 1,                # At 1, each iteration will be plotted, higher values will create an averging window for the specified iterations
    'color_bar': False,             # Plots a color bar for each heatmap
    'plot-opts': {
        'name-font-size': 12        # Font size
    }
}'''))
        nb.cells.append(nbf.v4.new_code_cell('''os.chdir("..")
evo_obj = wb.evolution.weEvolution(evolution_opts)
try:
    f, axarr = evo_obj.run()
except Exception as e:
    print(e)
os.chdir(curr_path)'''))
        nb.cells.append(nbf.v4.new_markdown_cell('''## Cluster Analysis
This analysis creates *pdist.h5*, which contains all of the probability distributions for each progress coordinate for each iteration. The cluster analysis attempts to find macrostates within the probability distribitions in your progress coordinates using scikit-learn's DBSCAN tool. These macrostates are recorded in *states.yaml* as a list of coordinates. While the coordinates are from the analysis histogram bins, the WESTPA bins that feature the coordinates will be considered part of that macrostate. If there is more than 1 progress coordinate, 2D scatter plots will be plotted to show the macrostates.'''))
        nb.cells.append(nbf.v4.new_code_cell('''cluster_opts = {
    **analsyis_opts,
    'threshold': 90,        # Percentile that the coordinate needs to reach to be considered part of a macrostate, or a dense coordinate
    'min-samples': 2,       # The minimum number of dense coordinates to be considered a macrostate
    'eps': 1.5              # The maximum difference 2 dense coordinates can have to be considered part of the same macrostate
}'''))
        nb.cells.append(nbf.v4.new_code_cell('''os.chdir("..")
clu_obj = wb.cluster.weCluster(cluster_opts)
try:
    figs = clu_obj.run()
except Exception as e:
    print(e)
os.chdir(curr_path)'''))
        nb.cells.append(nbf.v4.new_markdown_cell('''## Network Analysis
This analysis must be performed after the cluster analysis as it requires *states.yaml*. The network analysis traces the WESTPA simulation and calculates the rates from each macrostate alongside each macrostate's probabiltiy. The written results are recorded in *direct_output.txt*, and each rate as it evolves over the course of the simulation is plotted. It is important to note that all the rates are in units of "tau^-1". This means you must divide the rates by the specified tau of the WESTPA simulation in order to get the rates in inverse seconds.'''))
        nb.cells.append(nbf.v4.new_code_cell('''network_opts = {
    **analsyis_opts,
    'step-iter': 1          # Plots the rates by using traces from 1 to every Xth iteration
}'''))
        nb.cells.append(nbf.v4.new_code_cell('''os.chdir("..")
net_obj = wb.network.weNetwork(network_opts)
try:
    figs = net_obj.run()
except Exception as e:
    print(e)
os.chdir(curr_path)'''))
        nb.cells.append(nbf.v4.new_markdown_cell('''## Flux Analysis
This analysis calls `w_fluxanl` to compute the flux into the target state over 
the course of the simulation. It prints the average flux and rate with 95% 
confidence intervals, and plots the rate evolution over iterations with shaded 
confidence intervals. All values are in units of tau^-1. This analysis requires 
a target state to be defined (via `tstate.file` at `w_init` time) and works with 
any binning scheme.'''))
        nb.cells.append(nbf.v4.new_code_cell('''flux_opts = {
    **analsyis_opts,
    'step-iter': 1,         # Compute rate evolution every N iterations
    'output': 'flux.png',   # Name of the output figure
}'''))
        nb.cells.append(nbf.v4.new_code_cell('''os.chdir("..")
flux_obj = wb.flux.weFlux(flux_opts)
try:
    f, ax = flux_obj.run()
except Exception as e:
    print(e)
os.chdir(curr_path)'''))
        with open("analysis.ipynb", "w") as f:
            nbf.write(nb, f)

    def run(self):
        """
        runs the class functions in appropriate order to
        make the WESTPA simultion folder
        """
        self.make_sim_folders()
        if self.copy_run_net:
            self.copy_run_network()
        self.write_static_files()
        self.run_BNGL_on_file()
        self.sample_basis_states()
        self.write_dynamic_files()
        self.make_analysis_notebook()
        return
    