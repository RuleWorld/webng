import sys, os, json, hashlib, time
import numpy as np
import yaml
import subprocess as sbpc
import h5py
import webng.analysis.utils as utils


# Root attribute stamped onto every file a WESTPA tool builds for an analysis,
# recording what it was built from (see weAnalysis.stamp).
STAMP_ATTR = "webng_built_from"


class weAnalysis:
    """
    Base class for all analysis classes.
    """

    def __init__(self, opts):
        # Absolute path to the simulation folder. Everything (west.h5, system.py,
        # west.cfg) is found from here, so the analysis folder can live anywhere.
        self.sim_path = os.path.abspath(opts["sim_name"])
        self._use_this_sims_system_py()
        # keep opts around
        self.opts = opts
        # the simulation's data, opened read-only for the analysis's lifetime
        self.west_path = os.path.join(self.sim_path, "west.h5")
        self.h5file_path = self.west_path
        self.h5file = h5py.File(self.west_path, "r")
        # Set work path
        if self.opts["work-path"] is None:
            self.work_path = os.path.join(self.opts["sim_name"], "analysis")
        else:
            self.work_path = self.opts["work-path"]
        # we want to go there
        if not os.path.isdir(self.work_path):
            os.mkdir(self.work_path)
        self.curr_path = os.getcwd()
        # worker processes for WESTPA's tools (None: WESTPA's own default)
        self.n_workers = opts.get("n-workers")
        if self.n_workers is not None and (not isinstance(self.n_workers, int) or self.n_workers < 1):
            sys.exit("analyses: n-workers must be a whole number of at least 1, got {!r}".format(self.n_workers))
        # an existing assign.h5 to use as-is (path relative to where webng runs)
        provided = opts.get("assign-h5")
        self.provided_assign = os.path.abspath(provided) if provided else None
        self.assign_path = None        # set by ensure_assign
        os.chdir(self.work_path)
        # Set names if we have them
        self.set_names(opts["pcoords"])
        # iteration range (defaults: all finished iterations)
        self.first_iter, self.last_iter = self.set_iter_range(
            self._getd(opts, "first-iter", default=None, required=False),
            self._getd(opts, "last-iter", default=None, required=False),
        )

    def _use_this_sims_system_py(self):
        """
        Make `import system` (used to unpickle bin mappers, and by average's
        boundary plotting) resolve to THIS simulation's system.py. Each sim has
        its own, so when one Python process analyzes several sims (a notebook,
        or the test suite), the folder must go first on sys.path, and a
        `system` module already imported from a different sim must be dropped;
        otherwise the first sim's system.py silently wins for every later one.
        """
        if self.sim_path in sys.path:
            sys.path.remove(self.sim_path)
        sys.path.insert(0, self.sim_path)
        cached = sys.modules.get("system")
        cached_file = getattr(cached, "__file__", None)
        if cached is not None and (
            cached_file is None
            or os.path.dirname(os.path.abspath(cached_file)) != self.sim_path
        ):
            del sys.modules["system"]

    def _getd(self, dic, key, default=None, required=True):
        val = dic.get(key, default)
        if required and (val is None):
            sys.exit("{} is not specified in the dictionary".format(key))
        return val

    def set_dims(self, dims=None):
        """Number of pcoord dimensions (default: read from west.h5)."""
        if dims is None:
            dims = self.h5file["iterations/iter_{:08d}".format(1)]["pcoord"].shape[2]
        self.dims = dims
        return self.dims

    def _west_cfg_pcoord_names(self):
        """
        The pcoord names in the sim's west.cfg (west -> bng/librr -> data ->
        pcoords), or None. The propagator reads this list to decide what it
        writes, so it is exactly the order of west.h5's pcoord columns.
        """
        try:
            with open(os.path.join(self.sim_path, "west.cfg")) as f:
                cfg = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            return None
        west = cfg.get("west") or {}
        for section in ("bng", "librr"):
            names = ((west.get(section) or {}).get("data") or {}).get("pcoords")
            if names:
                return [str(n) for n in names]
        return None

    def set_names(self, names):
        """
        Names for west.h5's pcoord dimensions, by column. Prefers the sim's
        west.cfg, which records what the propagator actually wrote; the analysis
        options' pcoords list only matches if the sim was generated from those
        same options unchanged. If the count doesn't match west.h5, dimensions
        are labelled by number rather than guessed.
        """
        from_cfg = self._west_cfg_pcoord_names()
        if from_cfg is not None:
            if names is not None and list(names) != from_cfg:
                print("Using pcoord names from {} {} rather than the analysis "
                      "options' {}".format(os.path.join(self.sim_path, "west.cfg"),
                                           from_cfg, list(names)))
            names = from_cfg
        try:
            n_file = self.h5file["iterations/iter_{:08d}".format(1)]["pcoord"].shape[2]
        except (KeyError, AttributeError):
            n_file = None   # nothing to check against
        if names is not None and n_file is not None and len(names) != n_file:
            print("Warning: {} pcoord names given but west.h5 has {} pcoord "
                  "dimensions; labelling dimensions by number instead".format(
                      len(names), n_file))
            names = None
        if names is not None:
            self.names = dict(zip(range(len(names)), names))
        else:
            # We know the dimensionality, can assume a
            # naming scheme if we don't have one
            print("Giving default names to each dimension")
            dims = n_file or getattr(self, "dims", None) or self.set_dims()
            self.names = dict((i, str(i)) for i in range(dims))

    def set_iter_range(self, first_iter, last_iter):
        """Defaults: from iteration 1 through the last finished iteration."""
        if first_iter is None:
            first_iter = 1
        if last_iter is None:
            last_iter = self.h5file.attrs["west_current_iteration"] - 1
        return int(first_iter), int(last_iter)

    # ------------------------------------------------------------------
    # Cached outputs of WESTPA tools (pdist.h5, assign.h5, ...)
    #
    # Every analysis runs from its work path; west.h5 and the sim folder are
    # found through self.sim_path, so the work path can be anywhere. A tool's output is reused only if it was stamped as built from the same
    # inputs; otherwise it's rebuilt. That's what keeps e.g. a pdist.h5 from an
    # older or shorter run from silently being used for a new one.
    # ------------------------------------------------------------------

    def stamp(self, tool, **settings):
        """
        A string describing what an output was built from: which WESTPA tool,
        the identity of west.h5, and any settings that affect the result
        (iteration range, bins, ...).

        west.h5's identity is its current iteration, its size, and a
        fingerprint of the last finished iteration's walker records
        (seg_index: weights, parents, ...), so a continued, replaced, or
        different run all change it. Deliberately not its modification time,
        which changes without the run changing: copying the file, or a tool
        opening it writable (w_assign does) would make every cached output
        look stale.
        """
        with h5py.File(self.west_path, "r") as f:
            current = int(f.attrs["west_current_iteration"])
            last = "iterations/iter_{:08d}".format(current - 1)
            fingerprint = None
            if current > 1 and last in f and "seg_index" in f[last]:
                records = np.ascontiguousarray(f[last]["seg_index"][...])
                fingerprint = hashlib.sha256(records.tobytes()).hexdigest()
        built_from = {
            "tool": tool,
            "west_current_iteration": current,
            "west_size": os.path.getsize(self.west_path),
            "west_last_iteration": fingerprint,
        }
        built_from.update(settings)
        return json.dumps(built_from, sort_keys=True, default=str)

    @staticmethod
    def file_digest(path):
        """Content hash of a small input file (e.g. states.yaml)."""
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    @staticmethod
    def _stamp_changes(old, new):
        """What differs between two stamps, for the rebuild message."""
        if old is None:
            return "it has no stamp: made by an older WEBNG"
        try:
            old, new = json.loads(old), json.loads(new)
        except ValueError:
            return "unreadable stamp"
        changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
        return "changed: " + ", ".join(changed)

    @staticmethod
    def stamp_of(path):
        """The stamp recorded on an existing output, or None."""
        try:
            with h5py.File(path, "r") as f:
                value = f.attrs.get(STAMP_ATTR)
        except (OSError, KeyError):
            return None
        if isinstance(value, bytes):
            value = value.decode()
        return value

    def run_westpa_tool(self, output, command, stamp, cwd=None,
                        produced_at=None, log_path=None):
        """
        Make sure `output` (relative to the analysis folder) exists and was
        built from `stamp`, running `command` if not. `command` may also be a
        function returning the command, for commands that are costly to work
        out (it's only called if the tool actually has to run).

        cwd:         where to run the command (default: the analysis folder)
        produced_at: where the tool writes its file, if not directly at
                     `output` (it is moved into place afterwards)
        log_path:    if given, the tool's output is captured, echoed, and
                     saved here (progress lines ending in "..." are skipped)
        """
        if os.path.isfile(output) and self.stamp_of(output) == stamp:
            print("{} is up to date, reusing it".format(output))
            return
        if os.path.isfile(output) and self.stamp_of(output) is None:
            # not made by WEBNG (or by one from before stamping): it may have
            # taken hours to make, so never delete it on a guess
            raise RuntimeError(
                "{} in {} wasn't made by WEBNG (it has no WEBNG stamp), so WEBNG won't "
                "delete it. Delete it yourself to have WEBNG rebuild it; for an "
                "assign.h5 you made yourself, set analyses: assign-h5 to its path "
                "to use it as-is.".format(output, os.getcwd()))
        if callable(command):
            command = command()
        if os.path.isfile(output):
            print("{} was built from different inputs ({}); rebuilding it with {}".format(
                output, self._stamp_changes(self.stamp_of(output), stamp), command[0]))
            os.remove(output)
        else:
            print("{} does not exist. Running {}".format(output, command[0]))

        if log_path is None:
            result = sbpc.run(command, cwd=cwd)
        else:
            result = sbpc.run(command, cwd=cwd, stdout=sbpc.PIPE,
                              stderr=sbpc.STDOUT, text=True)
            with open(log_path, "w") as log:
                for line in result.stdout.splitlines(keepends=True):
                    print(line, end="")
                    if not line.strip().endswith("..."):
                        log.write(line)
        if result.returncode != 0:
            raise RuntimeError("{} failed with exit code {}; {} was not built"
                               .format(command[0], result.returncode, output))

        if produced_at is not None:
            os.replace(produced_at, output)
        if not os.path.isfile(output):
            raise RuntimeError("{} finished but {} was not created"
                               .format(command[0], output))
        with h5py.File(output, "a") as f:
            f.attrs[STAMP_ATTR] = stamp

    def load_run_mapper(self, n_iter=None, west_path=None):
        """
        The bin mapper this run used at iteration `n_iter` (default: its last
        finished iteration), from `west_path` (default: this analysis's
        west.h5). See webng.analysis.utils.load_mapper.
        """
        return utils.load_mapper(west_path or self.west_path, self.sim_path, n_iter)

    def pdist_bin_edges(self):
        """
        Histogram bin edges for every pcoord dimension over this analysis's
        iterations, as w_pdist takes them: at most `bins` bins per dimension.

        A dimension whose values are all whole numbers (molecule counts) gets
        bins of whole integers, edges on the half-integers: one bin per integer
        if its range fits in `bins`, otherwise the same number of integers in
        every bin. Equal-width bins narrower than 1, or not lined up with the
        integers, would leave some bins with no possible value and others with
        two, which shows up as a comb or ripple in every distribution.

        Any other dimension gets `bins` equal bins over its range, exactly as
        w_pdist makes them from a bin count.
        """
        lo = hi = whole = None
        first, last = int(self.first_iter), int(self.last_iter)
        n_total = last - first + 1
        report_every = max(1, n_total // 10)
        print("Scanning progress coordinates in iterations {}-{} to choose histogram "
              "bins...".format(first, last), flush=True)
        started = time.time()
        with h5py.File(self.west_path, "r") as f:
            for n_iter in range(first, last + 1):
                done = n_iter - first + 1
                if done % report_every == 0 or done == n_total:
                    print("  {}/{} iterations ({:.0f} s)".format(
                        done, n_total, time.time() - started), flush=True)
                pc = f["iterations/iter_{:08d}".format(n_iter)]["pcoord"][...]
                pc = pc.reshape(-1, pc.shape[-1])
                if pc.size == 0:
                    continue
                it_lo, it_hi = pc.min(axis=0), pc.max(axis=0)
                it_whole = np.all(pc == np.round(pc), axis=0)
                if lo is None:
                    lo, hi, whole = it_lo, it_hi, it_whole
                else:
                    lo, hi, whole = np.minimum(lo, it_lo), np.maximum(hi, it_hi), whole & it_whole
        if lo is None:
            raise RuntimeError("no progress coordinate data in iterations {} to {} of {}".format(
                self.first_iter, self.last_iter, self.west_path))
        edges = []
        for d in range(len(lo)):
            if whole[d]:
                first, last = int(round(lo[d])), int(round(hi[d]))
                n_values = last - first + 1
                per_bin = -(-n_values // self.bins)           # integers per bin (ceil)
                n_bins = -(-n_values // per_bin)
                edges.append([first - 0.5 + per_bin * i for i in range(n_bins + 1)])
            else:
                # as w_pdist does: nudge the top edge up so the maximum is inside
                lb, ub = float(lo[d]), float(hi[d] * 1.01 if hi[d] > 0 else hi[d] / 1.01)
                if ub <= lb:
                    lb, ub = lb - 0.5, lb + 0.5
                edges.append([float(x) for x in np.linspace(lb, ub, self.bins + 1)])
        return edges

    def pdist_stamp(self):
        """The stamp of a pdist.h5 built for this analysis. ("binning" names
        how edges are chosen, so files made the old way, from a plain bin
        count, are rebuilt instead of reused.)"""
        return self.stamp("w_pdist", first_iter=int(self.first_iter),
                          last_iter=int(self.last_iter), bins=self.bins,
                          binning="whole-integer-aware")

    def ensure_pdist(self):
        """
        pdist.h5 for this analysis's iteration range and bins, shared by
        average, evolution, and cluster. Returns it opened read-only. Its
        histograms cover first_iter..last_iter, in that order. Bins come from
        pdist_bin_edges (whole-number pcoords get whole-integer bins).
        """
        def command():          # the edges need a pass over west.h5: only if rebuilding
            return [
                "w_pdist",
                "-W", self.west_path,
                "--first-iter", "{}".format(self.first_iter),
                "--last-iter", "{}".format(self.last_iter),
                "-o", "pdist.h5",
                "-b", json.dumps(self.pdist_bin_edges()),
            ] + self.work_manager_args()
        self.run_westpa_tool("pdist.h5", command, self.pdist_stamp())
        return h5py.File("pdist.h5", "r")

    def ensure_assign(self):
        """
        assign.h5 from states.yaml, shared by network and pcca. Returns its
        stamp, which outputs built from it include in their own, and sets
        self.assign_path to the file to use.

        With the assign-h5 option, that existing file is used as-is and never
        rebuilt (e.g. assignments made with your own bins); its stamp then
        records which file it was and its contents' fingerprint, so outputs
        built from it are rebuilt if it's replaced.
        """
        if self.provided_assign is not None:
            path = self.provided_assign
            if not os.path.isfile(path):
                raise RuntimeError("assign-h5: {} does not exist".format(path))
            with h5py.File(path, "r") as f:
                if "state_map" not in f:
                    raise RuntimeError("assign-h5: {} has no state_map; is it w_assign "
                                       "output made with states?".format(path))
                fingerprint = hashlib.sha256()
                for name in ("state_map", "state_labels"):
                    if name in f:
                        fingerprint.update(np.ascontiguousarray(f[name][...]).tobytes())
            print("Using {} as-is (analyses: assign-h5); WEBNG won't rebuild it".format(path))
            self.assign_path = path
            return self.stamp("provided assign.h5", path=path, size=os.path.getsize(path),
                              contents=fingerprint.hexdigest())

        # runs from the sim folder (w_assign needs its west.cfg / system.py)
        self.assign_path = os.path.abspath("assign.h5")
        command = [
            "w_assign",
            "-W", self.west_path,
            "--states-from-file", os.path.abspath("states.yaml"),
            "-o", self.assign_path,
        ] + self.work_manager_args()
        stamp = self.stamp("w_assign", states_yaml=self.file_digest("states.yaml"))
        self.run_westpa_tool("assign.h5", command, stamp, cwd=self.sim_path)
        return stamp

    def work_manager_args(self):
        """WESTPA work-manager options for parallel tools (w_pdist, w_assign):
        n-workers worker processes, or nothing (WESTPA's default) if unset."""
        if self.n_workers is None:
            return []
        return ["--work-manager", "processes", "--n-workers", str(self.n_workers)]