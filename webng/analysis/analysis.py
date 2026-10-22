import sys, os, json, hashlib
import subprocess as sbpc
import h5py


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
        # get this in so we can import system.py if need be (e.g. unpickling bin mappers)
        sys.path.append(self.sim_path)
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
        os.chdir(self.work_path)
        # Set names if we have them
        self.set_names(opts["pcoords"])
        # iteration range (defaults: all finished iterations)
        self.first_iter, self.last_iter = self.set_iter_range(
            self._getd(opts, "first-iter", default=None, required=False),
            self._getd(opts, "last-iter", default=None, required=False),
        )

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

    def set_names(self, names):
        if names is not None:
            self.names = dict(zip(range(len(names)), names))
        else:
            # We know the dimensionality, can assume a
            # naming scheme if we don't have one
            print("Giving default names to each dimension")
            dims = getattr(self, "dims", None) or self.set_dims()
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
        the identity of west.h5 (its current iteration, size, and modification
        time, so a continued, replaced, or different run all change it), and
        any settings that affect the result (iteration range, bins, ...).
        """
        st = os.stat(self.west_path)
        with h5py.File(self.west_path, "r") as f:
            current = int(f.attrs["west_current_iteration"])
        built_from = {
            "tool": tool,
            "west_current_iteration": current,
            "west_size": st.st_size,
            "west_mtime_ns": st.st_mtime_ns,
        }
        built_from.update(settings)
        return json.dumps(built_from, sort_keys=True, default=str)

    @staticmethod
    def file_digest(path):
        """Content hash of a small input file (e.g. states.yaml)."""
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

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
        built from `stamp`, running `command` if not.

        cwd:         where to run the command (default: the analysis folder)
        produced_at: where the tool writes its file, if not directly at
                     `output` (it is moved into place afterwards)
        log_path:    if given, the tool's output is captured, echoed, and
                     saved here (progress lines ending in "..." are skipped)
        """
        if os.path.isfile(output):
            if self.stamp_of(output) == stamp:
                print("{} is up to date, reusing it".format(output))
                return
            print("{} was built from different inputs (or before stamping); "
                  "rebuilding it with {}".format(output, command[0]))
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

    def ensure_pdist(self):
        """
        pdist.h5 for this analysis's iteration range and bins, shared by
        average, evolution, and cluster. Returns it opened read-only. Its
        histograms cover first_iter..last_iter, in that order.
        """
        command = [
            "w_pdist",
            "-W", self.west_path,
            "--first-iter", "{}".format(self.first_iter),
            "--last-iter", "{}".format(self.last_iter),
            "-o", "pdist.h5",
            "-b", "{}".format(self.bins),
        ]
        stamp = self.stamp("w_pdist", first_iter=int(self.first_iter),
                           last_iter=int(self.last_iter), bins=self.bins)
        self.run_westpa_tool("pdist.h5", command, stamp)
        return h5py.File("pdist.h5", "r")

    def ensure_assign(self):
        """
        assign.h5 from states.yaml, shared by network and pcca. Returns its
        stamp, which outputs built from it include in their own.
        """
        # runs from the sim folder (w_assign needs its west.cfg / system.py)
        command = [
            "w_assign",
            "-W", self.west_path,
            "--states-from-file", os.path.abspath("states.yaml"),
            "-o", os.path.abspath("assign.h5"),
        ]
        stamp = self.stamp("w_assign", states_yaml=self.file_digest("states.yaml"))
        self.run_westpa_tool("assign.h5", command, stamp, cwd=self.sim_path)
        return stamp