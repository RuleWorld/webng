from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import yaml, os, shutil, sys, westpa, bionetgen, platform, stat, warnings, ctypes, random
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
        # replace an existing sim folder (webng setup --overwrite)
        self.overwrite = bool(getattr(args, "overwrite", False))
        self._parse_opts(self.opts)
        # TODO: make this optional somewhere else
        self.copy_run_net = True

    @staticmethod
    def _parse_wallclock(value):
        """
        sampling_options.max_wallclock -> "HH:MM:SS" for west.cfg's
        max_run_wallclock (WESTPA ends a run cleanly before exceeding it).
        Accepts "HH:MM:SS" text with any number of hours, or a number of
        seconds: YAML reads an unquoted 72:00:00 as a base-60 number of
        seconds (259200), so an options file may hand us either form.
        """
        if isinstance(value, bool):
            value = None
        if isinstance(value, (int, float)) and value > 0:
            seconds = int(value)
            return "{}:{:02d}:{:02d}".format(seconds // 3600, seconds % 3600 // 60, seconds % 60)
        text = str(value).strip()
        parts = text.split(":")
        if (len(parts) == 3 and all(p.isdigit() for p in parts)
                and int(parts[1]) < 60 and int(parts[2]) < 60 and any(int(p) for p in parts)):
            return "{}:{}:{}".format(int(parts[0]), parts[1].zfill(2), parts[2].zfill(2))
        sys.exit("sampling_options.max_wallclock must be HH:MM:SS (e.g. 72:00:00 "
                 "for 3 days, 720:00:00 for 30), got {!r}".format(value))

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
            propagator_options, "propagator_type", default="bng"
        )
        if self.propagator_type not in self.PROPAGATOR_TYPES:
            sys.exit("propagator_options.propagator_type must be one of {}, got {!r}".format(
                ", ".join(self.PROPAGATOR_TYPES), self.propagator_type))
        if self.propagator_type in ("libRoadRunner", "bng", "ezgillespz"):
            self.pcoord_list = self._getd(propagator_options, "pcoords")
        if self.propagator_type == "bng":
            self.libssa_path = self._getd(
                propagator_options, "libssa_path", default=None, required=False
            )
            self.gillespie_update_interval = self._getd(
                propagator_options, "gillespie_update_interval", default=1
            )
        if self.propagator_type == "ezgillespz":
            self._parse_ezgillespz_opts(propagator_options)

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
        self.max_wallclock = self._parse_wallclock(
            sampling_options.get("max_wallclock", "72:00:00"))

        # binning options
        binning_options = self._getd(self.opts, "binning_options")
        self.binning_style = self._getd(binning_options, "style", default='adaptive')
        if self.binning_style == 'adaptive':
            self.traj_per_bin = self._getd(binning_options, "traj_per_bin", default=10)
            self.block_size = self._getd(binning_options, "block_size", default=10)
            self.center_freq = self._getd(binning_options, "center_freq", default=1)
            self.max_centers = self._getd(binning_options, "max_centers", default=300)
        elif self.binning_style == "mabl":
            self._parse_mabl_opts(binning_options)
        else:
            self.boundaries   = self._getd(binning_options, "boundaries")
            self.traj_per_bin = self._getd(binning_options, "traj_per_bin", default=10)
            self.block_size   = self._getd(binning_options, "block_size",   default=10)
        if self.propagator_type == "ezgillespz":
            # The ezgillespz propagator runs each block of segments as one
            # batch. binning_options.block_size (segments per worker task,
            # 10 in every template) would cap batches at 10 walkers, so it
            # is not used: the batch size is propagator_options.block_size,
            # by default the whole iteration.
            self.block_size = self.ezg_block_size

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
    
    PROPAGATOR_TYPES = ("bng", "ezgillespz", "libRoadRunner", "executable")
    EZGILLESPZ_DEVICES = ("cpu", "gpu", "cuda", "metal")
    # larger than any iteration: WESTPA then hands the ezgillespz propagator
    # the whole iteration in one propagate() call
    EZGILLESPZ_BLOCK_SIZE = 1000000

    def _parse_ezgillespz_opts(self, propagator_options):
        """
        propagator_options for propagator_type: ezgillespz (all optional):
          device         cpu (default), gpu (CUDA or Metal, whichever the
                         machine has), cuda or metal
          kind           ezgillespz engine, default compact
          headroom_bits  fixed-point headroom, default 8
          thin           true: exact rate constants by thinning, default false
          seed           integer for a reproducible run, default none
          block_size     most walkers per batch (e.g. to fit GPU memory);
                         default: the whole iteration in one batch.
                         binning_options.block_size is not used.
        """
        device = str(self._getd(propagator_options, "device", default="cpu")).lower()
        if device not in self.EZGILLESPZ_DEVICES:
            sys.exit("propagator_options.device must be one of {}, got {!r}".format(
                ", ".join(self.EZGILLESPZ_DEVICES), device))
        self.ezg_device = device
        self.ezg_kind = self._getd(propagator_options, "kind", default="compact")
        self.ezg_headroom_bits = int(self._getd(propagator_options, "headroom_bits", default=8))
        self.ezg_thin = bool(self._getd(propagator_options, "thin", default=False))
        seed = self._getd(propagator_options, "seed", default=None, required=False)
        self.ezg_seed = None if seed is None else int(seed)
        block = self._getd(propagator_options, "block_size", default=None, required=False)
        self.ezg_block_size = self.EZGILLESPZ_BLOCK_SIZE if block is None else int(block)
        if self.ezg_block_size < 1:
            sys.exit("propagator_options.block_size must be a positive integer")

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

    def _parse_mabl_opts(self, binning_options):
        """
        binning_options for style: mabl. Lists are per progress coordinate,
        in pcoord order:
          n_walkers, n_split, weight_cap_factor, block_size
          start, target          microstates used for scoring
          coord_weights          importance weight per coordinate (0 = not scored)
          log_scale              true: fold-change progress (start, target > 0);
                                 a single true/false applies to every coordinate.
                                 Default false.
          merge_progress_floor   merge score = 1 / (W * max(S, floor)); default 0.01
          local_merge            pair merge candidates with their nearest
                                 neighbour instead of by rank; default false
          merge_pool             local merging: candidates = merge_pool x n_merge
                                 walkers by merge score; default 4 (at least 2)
          merge_log              write every merge to merge_dists.tsv (debugging);
                                 default false
        """
        dims = self.dims
        self.n_walkers = self._getd(binning_options, "n_walkers", default=40)
        self.n_split   = self._getd(binning_options, "n_split",   default=5)
        self.weight_cap_factor = self._getd(binning_options, "weight_cap_factor", default=10.0)
        self.start     = self._getd(binning_options, "start", default=[0]*dims)
        self.target    = self._getd(binning_options, "target", default=[50]*dims)
        self.block_size = self._getd(binning_options, "block_size", default=10)
        self.coord_weights = self._getd(binning_options, "coord_weights", default=[1]*dims)
        log_scale = self._getd(binning_options, "log_scale", default=False, required=False)
        if log_scale is None or isinstance(log_scale, bool):
            log_scale = [bool(log_scale)] * dims
        self.log_scale = [bool(x) for x in log_scale]
        self.merge_progress_floor = float(self._getd(
            binning_options, "merge_progress_floor", default=0.01, required=False))
        self.local_merge = bool(self._getd(binning_options, "local_merge", default=False, required=False))
        self.merge_pool = int(self._getd(binning_options, "merge_pool", default=4, required=False))
        self.merge_log = bool(self._getd(binning_options, "merge_log", default=False, required=False))
        self._check_mabl_opts()

    def _check_mabl_opts(self):
        """Errors out (before any files are written) on inconsistent MABL options."""
        for name in ("start", "target", "coord_weights", "log_scale"):
            value = getattr(self, name)
            if value is not None and len(value) != self.dims:
                sys.exit("binning_options.{} has {} entries; it needs one per progress "
                         "coordinate ({})".format(name, len(value), self.dims))
        names = list(getattr(self, "pcoord_list", None) or [])
        for d, use_log in enumerate(self.log_scale):
            if use_log and not (self.start[d] > 0 and self.target[d] > 0 and self.start[d] != self.target[d]):
                sys.exit("binning_options.log_scale is true for coordinate {} ({}), which needs "
                         "start and target > 0 and different (got start {}, target {})".format(
                             d, names[d] if d < len(names) else "-", self.start[d], self.target[d]))
        if self.merge_progress_floor < 0:
            sys.exit("binning_options.merge_progress_floor must be >= 0")
        if self.merge_pool < 2:
            sys.exit("binning_options.merge_pool must be at least 2")

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
        Returns dict: observable_name -> [(0-based species index, coefficient), ...].
        Every term in the group is kept (weighted sum), not just the first
        listed species — a group like "2*3,5*7" means 2*species3 + 5*species7.
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
                        obs_name = parts[1]
                        terms = []
                        for token in parts[2].split(","):
                            token = token.strip()
                            if "*" in token:
                                coeff_str, idx_str = token.split("*", 1)
                                coeff = float(coeff_str)
                            else:
                                idx_str, coeff = token, 1.0
                            try:
                                terms.append((int(idx_str) - 1, coeff))
                            except ValueError:
                                pass
                        if terms:
                            groups[obs_name] = terms
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
        Filter frames from a long SSA trajectory that satisfy basis_region
        conditions (evaluated on pcoords using the groups mapping from
        init.net), and write:

          bstates/microstate_pool.dat  — one line per valid frame: <weight> <sp1> <sp2> ... <spN>
          bstates/0.net                — placeholder required by w_init
          bstates/bstates.txt          — single entry: 0 1 0.net

        If recycling is disabled, copies init.net → 0.net as before and
        writes the single-entry bstates.txt.

        The sampling trajectory itself is run with whichever engine the
        main propagator already uses — the bng shared library if
        propagator_type == "bng" (fast, no libRoadRunner/SBML needed),
        otherwise libRoadRunner (used for both the "libRoadRunner" and
        "executable" propagator types, matching prior behavior).
        """
        bstates_dir = "bstates"
        init_net    = "bngl_conf/init.net"

        # no recycle: single basis state
        if not self.do_recycling:
            shutil.copyfile(init_net, os.path.join(bstates_dir, "0.net"))
            self._write_bstatestxt(n=1)
            return

        # validate pcoords against net files (shared by both engines)
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

        if self.propagator_type == "bng":
            self._sample_basis_states_bng(init_net, groups, n_species)
        elif self.propagator_type == "ezgillespz":
            self._sample_basis_states_ezgillespz(init_net, groups, n_species)
        else:
            self._sample_basis_states_libRR(init_net, groups, n_species)

    def _sample_basis_states_bng(self, init_net, groups, n_species):
        """
        Basis-state sampling using the bng shared library (libssa.so)
        directly via ctypes — no init.xml/SBML export needed. Used when
        propagator_type == "bng", so users running with bng don't need a
        libRoadRunner install or a redundant XML export just for this
        one-time setup step.
        """
        lib = self._load_bng_lib_for_sampling()
        seed = random.randint(0, 2**14)
        status = lib.ssa_init(
            init_net.encode(), seed, self.gillespie_update_interval
        )
        if status != 0:
            err = lib.ssa_get_last_error().decode()
            sys.exit("[recycling] ssa_init failed: {}".format(err))
        lib_n_species = lib.ssa_get_n_species()
        if lib_n_species != n_species:
            sys.exit(
                "[recycling] species count mismatch: .net file has {} but "
                "bng library reports {}.".format(n_species, lib_n_species)
            )

        print("[recycling] Running sampling trajectory ({} steps) via bng …".format(
            self.bs_traj_length))
        num_ts = self.bs_traj_length + 1
        buf = (ctypes.c_double * (num_ts * n_species))()
        status = lib.ssa_run_multi(0.0, float(self.bs_traj_length), num_ts, buf, n_species)
        if status != 0:
            err = lib.ssa_get_last_error().decode()
            sys.exit("[recycling] ssa_run_multi failed: {}".format(err))
        all_states = np.ctypeslib.as_array(buf).reshape(num_ts, n_species)
        self._write_microstate_pool(all_states, init_net, groups, n_species)

    @staticmethod
    def _ezgillespz_module():
        """
        The ezgillespz propagator module bundled with WEBNG
        (webng/assets/ezgillespz_propagator.py), imported from its file:
        setup-time sampling uses the same engine handling as the run.
        """
        import importlib.util
        path = os.path.join(weConvert._assets_dir(), "ezgillespz_propagator.py")
        spec = importlib.util.spec_from_file_location("webng_ezgillespz_propagator", path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except ImportError as e:
            sys.exit("[ezgillespz] {}".format(e))
        return mod

    def _sample_basis_states_ezgillespz(self, init_net, groups, n_species):
        """
        Basis-state sampling with ezgillespz on the CPU: one walker from
        init.net, recorded every time unit for traj_length time units (the
        same frames as the bng sampler), then filtered by basis_region.
        """
        mod = self._ezgillespz_module()
        model = mod.gg.load_net(init_net)
        if model.ns != n_species:
            sys.exit(
                "[recycling] species count mismatch: .net file has {} but "
                "ezgillespz reads {}.".format(n_species, model.ns)
            )
        runner = mod.Runner(model, device="cpu", kind=self.ezg_kind,
                            headroom_bits=self.ezg_headroom_bits, thin=self.ezg_thin)
        rng = np.random.default_rng(self.ezg_seed)
        print("[recycling] Running sampling trajectory ({} steps) via ezgillespz …".format(
            self.bs_traj_length))
        num_ts = self.bs_traj_length + 1
        all_states = np.empty((num_ts, n_species), dtype=np.float64)
        X = mod.as_counts(model.x0, n_species, init_net)[None, :]
        all_states[0] = X[0]
        for t in range(1, num_ts):
            X, _ = runner.run(X, int(rng.integers(1, 2 ** 62)), 1.0)
            all_states[t] = X[0]
        self._write_microstate_pool(all_states, init_net, groups, n_species)

    def _write_microstate_pool(self, all_states, init_net, groups, n_species):
        """
        Keep the sampled frames (rows of all_states: species counts) that
        satisfy basis_region and write them to bstates/microstate_pool.dat,
        with the placeholder 0.net and bstates.txt w_init needs. Shared by
        the bng and ezgillespz samplers.
        """
        num_ts = len(all_states)
        bstates_dir = "bstates"

        # compute each basis_region observable as the weighted sum of its
        # group's species terms, across every sampled frame
        pcoord_cols = {}
        for cond in self.basis_region:
            obs   = cond["observable"]
            terms = groups[obs]
            pcoord_cols[obs] = sum(coeff * all_states[:, idx] for idx, coeff in terms)

        mask = np.ones(num_ts, dtype=bool)
        for cond in self.basis_region:
            col = pcoord_cols[cond["observable"]]
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
        pool_path = os.path.join(bstates_dir, "microstate_pool.dat")
        # Weight = 1.0 for all frames (uniform over valid frames).
        # TODO Future enhancement: weight by dwell time for equilibrium sampling.
        with open(pool_path, "w") as pf:
            pf.write("# weight " +
                     " ".join("sp{}".format(i+1) for i in range(n_species)) + "\n")
            for idx in valid_idx:
                counts = all_states[idx]
                count_strs = " ".join(
                    ("{:.0f}" if c == int(c) else "{:.6g}").format(c) for c in counts
                )
                pf.write("1.0 {}\n".format(count_strs))
        print("[recycling] Wrote {} microstates to {}.".format(n_valid, pool_path))
        for cond in self.basis_region:
            col = pcoord_cols[cond["observable"]][valid_idx]
            print("[recycling]   {}: min={:.1f}  max={:.1f}  mean={:.1f}".format(
                cond["observable"], col.min(), col.max(), col.mean()))
        shutil.copyfile(init_net, os.path.join(bstates_dir, "0.net"))
        self._write_bstatestxt(n=1)

    def _sample_basis_states_libRR(self, init_net, groups, n_species):
        """
        Basis-state sampling using libRoadRunner. Used for propagator_type
        in {"libRoadRunner", "executable"}. Observable values come from
        RoadRunner's own SBML-based evaluation (rr.timeCourseSelections),
        so — unlike the bng path — no manual group-weighting is needed here;
        `groups` is accepted for signature symmetry but unused.
        """
        init_xml = "bngl_conf/init.xml"
        bstates_dir = "bstates"
        try:
            import roadrunner as librr
        except ImportError:
            sys.exit("[recycling] libRoadRunner is required for basis-state "
                     "sampling but is not installed.")
        print("[recycling] Running sampling trajectory ({} steps) via libRoadRunner …".format(
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

    @staticmethod
    def _assets_dir():
        """webng/assets: files copied into simulation folders as they are
        (propagators, plugins, scripts) and the bundled libssa builds."""
        core_dir = os.path.dirname(os.path.abspath(__file__))   # webng/core
        return os.path.join(os.path.dirname(core_dir), "assets")  # webng/assets

    def _copy_asset(self, name, dest=None, mode=None):
        """
        Copy webng/assets/<name> into the simulation folder (the current
        folder during setup) as dest, by default under the same name, and
        set its permissions to mode if given. For the files that are the same
        for every simulation; files that depend on the options are still
        written by their _write_* methods.
        """
        dest = dest or name
        shutil.copyfile(os.path.join(self._assets_dir(), name), dest)
        if mode is not None:
            os.chmod(dest, mode)

    def _write_librrPropagator(self):
        """libRR_propagator.py, the same for every simulation: copied from webng/assets."""
        self._copy_asset("libRR_propagator.py")

    def _write_bngPropagator(self):
        """
        bng_propagator.py, the same for every simulation: copied from
        webng/assets. It loads libssa.so (see _copy_bng_lib) from its own
        folder and reads its settings from west.cfg's west.bng section.
        """
        self._copy_asset("bng_propagator.py")

    def _write_ezgillespzPropagator(self):
        """
        ezgillespz_propagator.py, the same for every simulation: copied from
        webng/assets. Its settings come from west.cfg's west.ezgillespz
        section. ezgillespz itself must be installed where w_run runs; it is
        not a WEBNG dependency.
        """
        self._copy_asset("ezgillespz_propagator.py")
        try:
            import ezgillespz  # noqa: F401
        except ImportError:
            warnings.warn(
                "propagator_type is ezgillespz, but ezgillespz is not installed in "
                "this environment. Install it (pip install -e /path/to/ezgillespz) "
                "in the environment that will run w_run.")

    @staticmethod
    def _bng_lib_name(system=None, machine=None):
        """
        The bundled libssa build for a platform, by name:
        libssa-<os>-<cpu>.<so|dylib>, where <cpu> is what platform.machine()
        reports (e.g. libssa-linux-x86_64.so, libssa-macos-arm64.dylib).
        build_libssa.sh names its output the same way. Defaults to this machine.
        """
        system = system or platform.system()
        machine = machine or platform.machine()
        os_name, ext = {"Linux": ("linux", "so"), "Darwin": ("macos", "dylib")}.get(
            system, (system.lower(), "so"))
        return "libssa-{}-{}.{}".format(os_name, machine, ext)

    def _resolve_bng_lib_path(self):
        """
        Where to find libssa:
          1. propagator_options.libssa_path, if the user set one (e.g. a
             custom local build), or
          2. the build bundled with the WEBNG package for this platform,
             webng/assets/libssa-<os>-<cpu>.<so|dylib> (see _bng_lib_name).
        """
        if getattr(self, "libssa_path", None):
            return self.libssa_path
        return os.path.join(self._assets_dir(), self._bng_lib_name())

    def _copy_bng_lib(self):
        """
        Copies libssa into the simulation root as libssa.so, alongside
        bng_propagator.py, which loads it by that name relative to its own
        __file__ (the .so name works for a macOS build too; ctypes doesn't
        care about the extension).
        """
        src = self._resolve_bng_lib_path()
        if not os.path.isfile(src):
            assets = self._assets_dir()
            bundled = sorted(f for f in os.listdir(assets) if f.startswith("libssa-")) \
                if os.path.isdir(assets) else []
            hint = ("On Windows, run WEBNG under WSL, where the Linux build works. "
                    if platform.system() == "Windows" else "")
            sys.exit(
                "No libssa build for this platform ({}-{}): expected '{}'.\n"
                "Builds bundled with this WEBNG: {}.\n"
                "{}Build one for this machine with libssa/build_libssa.sh, then "
                "copy it into webng/assets/ or set propagator_options.libssa_path "
                "to it.".format(platform.system(), platform.machine(), src,
                                ", ".join(bundled) or "none", hint)
            )
        shutil.copyfile(src, "libssa.so")

    def _load_bng_lib_for_sampling(self):
        """
        Loads libssa.so directly into this (weConvert) process via ctypes,
        for one-time basis-state sampling at setup time. This is separate
        from the copy written into the sim folder for the per-segment
        propagator to use at run time — same resolution rule, same file.
        """
        lib_path = self._resolve_bng_lib_path()
        if not os.path.isfile(lib_path):
            sys.exit(
                "[recycling] No libssa build for this platform ({}-{}): "
                "expected '{}'.\nBuild one for this machine with "
                "libssa/build_libssa.sh, then copy it into webng/assets/ or set "
                "propagator_options.libssa_path to it.".format(
                    platform.system(), platform.machine(), lib_path)
            )
        try:
            lib = ctypes.CDLL(lib_path)
        except OSError as e:
            sys.exit(
                "[recycling] Found libssa.so at '{}' but could not load it "
                "({}).\nThis is almost always a platform/architecture "
                "mismatch — e.g. a Linux-built libssa.so on macOS or "
                "Windows. Rebuild it for this machine with build_libssa.sh "
                "and, if needed, point propagator_options.libssa_path at "
                "the result.".format(lib_path, e)
            )

        lib.ssa_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.ssa_init.restype  = ctypes.c_int

        lib.ssa_run_multi.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ]
        lib.ssa_run_multi.restype = ctypes.c_int

        lib.ssa_get_n_species.argtypes = []
        lib.ssa_get_n_species.restype  = ctypes.c_int

        lib.ssa_get_last_error.argtypes = []
        lib.ssa_get_last_error.restype  = ctypes.c_char_p

        return lib

    def _write_restartDriver(self):
        """restart_plugin.py, the same for every simulation: copied from webng/assets."""
        self._copy_asset("restart_plugin.py")


    MABL_CONFIG_BEGIN = "# >>> MABL configuration"
    MABL_CONFIG_END = "# <<< end MABL configuration"

    def _mabl_config_block(self):
        """The settings block of MABL_driver.py, from binning_options and recycling."""
        names = list(getattr(self, "pcoord_list", None) or ["d%d" % i for i in range(self.dims)])
        return "\n".join([
            self.MABL_CONFIG_BEGIN + " (written by webng setup from binning_options and recycling)",
            "# Progress coordinates, in order (west.cfg); the lists below follow this order.",
            "MABL_PCOORDS = {!r}".format(names),
            "",
            "# Start and target microstate per coordinate (used for scoring only).",
            "MABL_START = {}".format(self.start),
            "MABL_TARGET = {}".format(self.target),
            "",
            "# Per-coordinate importance weights (0 = not scored).",
            "MABL_COORD_WEIGHTS = {}".format(self.coord_weights),
            "",
            "# Per-coordinate progress scale: False linear, True log (fold change).",
            "MABL_LOG_SCALE = {}".format(self.log_scale),
            "",
            "# Walkers split per iteration (an equal number of pairwise merges follows).",
            "MABL_N_SPLIT = {}".format(self.n_split),
            "",
            "# Split any walker heavier than this factor x the average weight.",
            "MABL_WEIGHT_CAP_FACTOR = {}".format(self.weight_cap_factor),
            "",
            "# Merge score = 1 / (W * max(S, floor)).",
            "MABL_MERGE_PROGRESS_FLOOR = {}".format(self.merge_progress_floor),
            "",
            "# Local merging: pair candidates with their nearest neighbour among the",
            "# top MABL_MERGE_POOL x n_merge walkers by merge score.",
            "MABL_LOCAL_MERGE = {}".format(self.local_merge),
            "MABL_MERGE_POOL = {}".format(self.merge_pool),
            "",
            "# Debugging: log every pairwise merge to merge_dists.tsv.",
            "MABL_MERGE_LOG = {}".format(self.merge_log),
            "",
            "# Recycling region: [min, max] per coordinate, in pcoord order; None = no bound.",
            "MABL_RECYCLE_REGION = {}".format(self.recycle_region),
            self.MABL_CONFIG_END,
        ])

    def _write_mabl_driver(self):
        """
        MABL_driver.py: webng/assets/MABL_driver.py with its configuration
        block (between the MABL_CONFIG_BEGIN and MABL_CONFIG_END lines)
        replaced by this config's settings (_mabl_config_block). Everything
        else is the asset as it is.
        """
        with open(os.path.join(self._assets_dir(), "MABL_driver.py")) as f:
            text = f.read()
        i = text.index(self.MABL_CONFIG_BEGIN)
        j = text.index(self.MABL_CONFIG_END, i) + len(self.MABL_CONFIG_END)
        with open("MABL_driver.py", "w") as f:
            f.write(text[:i] + self._mabl_config_block() + text[j:])

    def _write_runsh(self):
        """
        write the run.sh file for WESTPA simulations
        """
        # TODO: Add submission scripts for varied clusters
        # TODO: Add a hook to write any submission scripts?
        if self.propagator_type == "ezgillespz":
            # ezgillespz parallelizes inside each batch (all CPU threads, or
            # the GPU), so one process runs the whole iteration: no segments
            # pickled between processes, and one GPU context
            lines = ["#!/bin/bash", 'w_run --work-manager serial "$@"']
        else:
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
        aux_functions.py (the executable propagator's pcoord loader, which
        skips the time column of BNG output), the same for every simulation:
        copied from webng/assets.
        """
        self._copy_asset("aux_functions.py")

    def _write_bstatestxt(self, n=1):
        lines = ["{} 1 {}.net\n".format(i, i) for i in range(n)]
        with open("bstates/bstates.txt", "w") as f:
            f.writelines(lines)

    def _write_getpcoord_sh(self):
        """
        westpa_scripts/get_pcoord.sh, the pcoord script for the initial
        center, the same for every simulation: copied from webng/assets.
        """
        # OSDEPEND: EXECUTABLE - This won't work on Windows powershell
        self._copy_asset("get_pcoord.sh", "westpa_scripts/get_pcoord.sh", mode=0o764)

    def _write_postiter_sh(self):
        """
        westpa_scripts/post_iter.sh, a post-iteration script that deletes
        iterations older than 3 iterations, the same for every simulation:
        copied from webng/assets.
        """
        # OSDEPEND: EXECUTABLE - This won't work on Windows powershell
        self._copy_asset("post_iter.sh", "westpa_scripts/post_iter.sh", mode=0o764)

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
        if self.propagator_type == "executable":
            self._executable_westcfg()
        elif self.propagator_type == "libRoadRunner":
            self._libRR_westcfg()
        elif self.propagator_type == "bng":
            self._bng_westcfg()
        elif self.propagator_type == "ezgillespz":
            self._ezgillespz_westcfg()

    def _ezgillespz_westcfg(self):
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
        seed = "null" if self.ezg_seed is None else str(self.ezg_seed)

        lines = (
            ["# vi: set filetype=yaml :","---","west:"]
            + drivers_block
            + [
            "  system:",
            "    driver: system.System",
            "    module_path: $WEST_SIM_ROOT",
            "  propagation:",
            "    max_total_iterations: {}".format(self.max_iter),
            "    max_run_wallclock:    {}".format(self.max_wallclock),
            "    propagator:           ezgillespz_propagator.ezgillespzPropagator",
            "    gen_istates:          {}".format(gen_istates),
            "    # segments per propagate() call; the propagator runs each call as one batch",
            "    block_size:           {}".format(self.block_size),
            "  data:",
            "    west_data_file: west.h5",
            "    datasets:",
            "      - name:        pcoord",
            "        scaleoffset: 4",
            "      - name:        final_state",
            "        scaleoffset: 4",
            "      - name:        pool_microstate",
            "        scaleoffset: 4",
            "  plugins:"]
            + plugin_insert
            + ["    - plugin: restart_plugin.RestartDriver",
            "  ezgillespz:",
            "    init:",
            "      net_file: ./bngl_conf/init.net",
            "      init_time_step: 0",
            "      final_time_step: {}".format(self.tau),
            "      num_time_step: {}".format(self.plen + 1),
            "      device: {}".format(self.ezg_device),
            "      kind: {}".format(self.ezg_kind),
            "      headroom_bits: {}".format(self.ezg_headroom_bits),
            "      thin: {}".format("true" if self.ezg_thin else "false"),
            "      seed: {}".format(seed),
            "    data:",
            "      pcoords: {}".format(('["' + '","'.join(self.pcoord_list) + '"]')),
            ]
        )

        full_text = "\n".join(lines)
        with open("west.cfg", "w") as f:
            f.write(full_text)

    def _bng_westcfg(self):
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
            "    max_run_wallclock:    {}".format(self.max_wallclock),
            "    propagator:           bng_propagator.bngPropagator",
            "    gen_istates:          {}".format(gen_istates),
            "    block_size:           {}".format(self.block_size),
            "  data:\n",
            "    west_data_file: west.h5",
            "    datasets:",
            "      - name:        pcoord",
            "        scaleoffset: 4",
            "      - name:        final_state",
            "        scaleoffset: 4",
            "      - name:        pool_microstate",
            "        scaleoffset: 4",
            "  plugins:"]
            + plugin_insert
            + ["    - plugin: restart_plugin.RestartDriver",
            "  bng:",
            "    init:",
            "      net_file: ./bngl_conf/init.net",
            "      init_time_step: 0",
            "      final_time_step: {}".format(self.tau),
            "      num_time_step: {}".format(step_no + 1),
            "      gillespie_update_interval: {}".format(self.gillespie_update_interval),
            "    data:",
            "      pcoords: {}".format(('["' + '","'.join(self.pcoord_list) + '"]')),
            ]
        )

        full_text = "\n".join(lines)
        with open("west.cfg", "w") as f:
            f.write(full_text)

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
            "    max_run_wallclock:    {}".format(self.max_wallclock),
            "    propagator:           libRR_propagator.librrPropagator ",
            "    gen_istates:          {}".format(gen_istates),
            "    block_size:           {}".format(self.block_size),
            "  data:\n",
            "    west_data_file: west.h5",
            "    datasets:",
            "      - name:        pcoord",
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
            "    max_run_wallclock:    {}".format(self.max_wallclock),
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
            "  RUNNET_STATUS=$?",
            '  if [[ $RUNNET_STATUS -ne 0 ]]; then',
            '    echo "runseg.sh: run_network exited with status $RUNNET_STATUS (CONTINUES)" >&2',
            "    exit $RUNNET_STATUS",
            "  fi",
            "  if [[ ! -f seg.gdat ]]; then",
            "    DATA_LINES=0",
            "  else",
            "    DATA_LINES=$(grep -vc '^#' seg.gdat)",
            "  fi",
            '  if [[ ! -s seg.gdat ]] || [[ "$DATA_LINES" -eq 0 ]]; then',
            '    echo "runseg.sh: seg.gdat has no usable data rows after run_network (CONTINUES). '
            'size=$(wc -c < seg.gdat 2>/dev/null || echo missing) non_comment_lines=$DATA_LINES" >&2',
            '    echo "--- seg.gdat contents ---" >&2',
            "    cat seg.gdat >&2 2>/dev/null || true",
            '    echo "--- end seg.gdat ---" >&2',
            "    exit 1",
            "  fi",
            "  tail -n 1 parent.gdat > $WEST_PCOORD_RETURN",
            "  cat seg.gdat >> $WEST_PCOORD_RETURN",
            '  echo "runseg.sh: wrote to WEST_PCOORD_RETURN=$WEST_PCOORD_RETURN '
            'size=$(wc -c < "$WEST_PCOORD_RETURN" 2>/dev/null || echo missing) '
            'lines=$(wc -l < "$WEST_PCOORD_RETURN" 2>/dev/null || echo missing) (CONTINUES)" >&2',
            'elif [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_NEWTRAJ" ]; then',
            "  if [[ -n $SCRATCH ]];then",
            "    cp $WEST_PARENT_DATA_REF ./parent.net",
            "  else",
            "    ln -sv $WEST_PARENT_DATA_REF ./parent.net",
            "  fi",
            "  $RunNet -o ./seg -p ssa -h $WEST_RAND16 --cdat 0 --fdat 0 -e -g ./parent.net ./parent.net {} {}".format(
                step_len, step_no
            ),
            "  RUNNET_STATUS=$?",
            '  if [[ $RUNNET_STATUS -ne 0 ]]; then',
            '    echo "runseg.sh: run_network exited with status $RUNNET_STATUS (NEWTRAJ)" >&2',
            "    exit $RUNNET_STATUS",
            "  fi",
            "  if [[ ! -f seg.gdat ]]; then",
            "    DATA_LINES=0",
            "  else",
            "    DATA_LINES=$(grep -vc '^#' seg.gdat)",
            "  fi",
            '  if [[ ! -s seg.gdat ]] || [[ "$DATA_LINES" -eq 0 ]]; then',
            '    echo "runseg.sh: seg.gdat has no usable data rows after run_network (NEWTRAJ). '
            'size=$(wc -c < seg.gdat 2>/dev/null || echo missing) non_comment_lines=$DATA_LINES" >&2',
            '    echo "--- seg.gdat contents ---" >&2',
            "    cat seg.gdat >&2 2>/dev/null || true",
            '    echo "--- end seg.gdat ---" >&2',
            "    exit 1",
            "  fi",
            "  cat seg.gdat > $WEST_PCOORD_RETURN",
            '  echo "runseg.sh: wrote to WEST_PCOORD_RETURN=$WEST_PCOORD_RETURN '
            'size=$(wc -c < "$WEST_PCOORD_RETURN" 2>/dev/null || echo missing) '
            'lines=$(wc -l < "$WEST_PCOORD_RETURN" 2>/dev/null || echo missing) (NEWTRAJ)" >&2',
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
        elif self.propagator_type == "bng":
            self._write_restartDriver()
            self._write_bngPropagator()
            self._copy_bng_lib()
        elif self.propagator_type == "ezgillespz":
            self._write_restartDriver()
            self._write_ezgillespzPropagator()
        if self.binning_style == "mabl":
            self._write_mabl_driver()
        if self.do_recycling:
            self._write_tstates()

    def make_sim_folders(self):
        """
        make folders WESTPA needs
        """
        self.sim_dir = self.fname
        if os.path.exists(self.fname):
            if not getattr(self, "overwrite", False):
                sys.exit(
                    "The simulation folder '{}' already exists. Choose another "
                    "path_options.sim_name, delete the folder, or run "
                    "`webng setup --overwrite` to replace it (which deletes "
                    "everything in it, including any west.h5).".format(self.fname))
            # never delete the folder we're in, or one above it
            target, here = os.path.realpath(self.fname), os.path.realpath(os.getcwd())
            if target == here or here.startswith(target + os.sep):
                sys.exit("Refusing to overwrite '{}': it is the current folder or "
                         "contains it. Set path_options.sim_name to a subfolder."
                         .format(self.fname))
            print("Overwriting the existing simulation folder '{}'".format(self.fname))
            shutil.rmtree(self.fname)
        os.makedirs(self.fname)
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
        elif self.propagator_type in ("bng", "ezgillespz"):
            # bng and ezgillespz only ever need the .net file — both the
            # propagators and their basis-state samplers (used for
            # recycling) read the .net file directly, with no SBML/XML
            # export step. (libRoadRunner is only needed for recycling
            # when propagator_type == "libRoadRunner"; see
            # _sample_basis_states_libRR.)
            self._executable_BNGL_on_file()

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
The cluster analysis finds macrostates and records them in *states.yaml* as lists of coordinates; every WESTPA bin containing one of a state's coordinates belongs to that state. The network analysis below then uses *states.yaml*. Two methods:

* **dbscan** (where the walkers spend their time): keeps the most probable WESTPA bins until they hold `probability-mass` of the probability, then groups the kept bins that lie close together with scikit-learn's DBSCAN. Distances are measured with every progress coordinate scaled to 0-1 over the range the run covered (after log10(x + 1) with `log-scale`), so `eps` is a fraction of that range, or chosen automatically. Its states are the probable regions; other bins belong to no state, and states holding little probability are cut first. It also writes *cluster_dbscan.txt*.
* **gpcca** (dynamics): builds the run's bin-to-bin transition matrix (lag tau) and splits the bins into metastable states with GPCCA+, so each state is a set of bins the walkers move among quickly and leave slowly. Every bin the walkers move back and forth through gets a state, and states holding little probability (rare phenotypes) are kept. It also writes *cluster_gpcca.txt* with each state's stationary probability and the state-to-state transition probabilities per iteration.

If there is more than 1 progress coordinate, 2D scatter plots show the macrostates.'''))
        nb.cells.append(nbf.v4.new_code_cell('''cluster_opts = {
    **analsyis_opts,
    'method': 'dbscan',     # 'dbscan' (density) or 'gpcca' (dynamics)
    # dbscan
    'probability-mass': 0.9,  # Keep the most probable bins until they hold this fraction of the probability
    'eps': None,            # How close (fraction 0-1 of each coordinate's range) two kept bins must be to be grouped. 'None' chooses it automatically
    'min-samples': 2,       # Bins (itself included) a bin needs within eps to start a macrostate
    'log-scale': False,     # Measure distances on log10(x + 1), for coordinates spanning orders of magnitude
    # gpcca
    'n-clusters': None,     # Number of macrostates. 'None' lets GPCCA+ choose between 2 and max-clusters by crispness
    'max-clusters': 10,
    'assign-h5': None,      # Optional w_assign file to take the bins from (e.g. someone else's bins). 'None' uses the run's own
}'''))
        nb.cells.append(nbf.v4.new_code_cell('''os.chdir("..")
clu_obj = wb.cluster.weCluster(cluster_opts)
try:
    figs = clu_obj.run()
except Exception as e:
    print(e)
os.chdir(curr_path)'''))
        nb.cells.append(nbf.v4.new_markdown_cell('''## Network Analysis
This analysis must be performed after the cluster analysis as it requires *states.yaml* (from either cluster method, or written by hand). It assigns the run's walkers to those states (`w_assign`) and estimates the rates and mean first passage times (MFPTs) between them two ways:

* **direct** (`w_direct`): each walker is labelled with the state it visited last; the rate from A to B is the flux of A-labelled walkers into B divided by the A-labelled probability. Rates are per tau ("tau^-1"); each rate's evolution is plotted (*rate_A_to_B.png*), and the tool's log is in *direct_output.txt*.
* **matrix**: the bin-to-bin transition matrix at lag tau (each segment's weight counted from the bin it started in to the bin it ended in), lumped into the states (each state's stationary probability and the state-to-state transition probabilities), and MFPTs from it, plotted as they evolve (*mfpt_evolution.png*). Set `'matrix': False` to skip.

Both are summarized in *network.txt*, with MFPTs in real time (iterations * tau). They answer slightly different questions and need not agree exactly; for a precise rate of one transition, a run targeted at it is the better estimate.'''))
        nb.cells.append(nbf.v4.new_code_cell('''network_opts = {
    **analsyis_opts,
    'tau': ''' + repr(self.tau) + ''',             # real time per WE iteration (sampling_options.tau)
    'step-iter': 1,         # Evaluate the rates and MFPTs every N iterations
    'matrix': True,         # Also the transition matrix and its MFPTs
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
        # Setup works from inside the sim folder (make_sim_folders changes into
        # it), so return to where we started when done, even if a step fails.
        # Harmless from the command line, but a Python caller (a notebook, a
        # script, the tests) would otherwise be left inside the sim folder.
        start_dir = os.getcwd()
        try:
            self.make_sim_folders()
            if self.copy_run_net:
                self.copy_run_network()
            self.write_static_files()
            self.run_BNGL_on_file()
            self.sample_basis_states()
            self.write_dynamic_files()
            self.make_analysis_notebook()
        finally:
            os.chdir(start_dir)
        return