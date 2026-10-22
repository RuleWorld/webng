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
        # get this in so we can import system.py if need be. Absolute, because
        # we chdir into the analysis folder below and a relative entry would
        # then point at the wrong place (breaking e.g. unpickling bin mappers).
        sys.path.append(os.path.abspath(opts["sim_name"]))
        # keep opts around
        self.opts = opts
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

    def _getd(self, dic, key, default=None, required=True):
        val = dic.get(key, default)
        if required and (val is None):
            sys.exit("{} is not specified in the dictionary".format(key))
        return val

    def set_names(self, names):
        if names is not None:
            self.names = dict(zip(range(len(names)), names))
        else:
            # We know the dimensionality, can assume a
            # naming scheme if we don't have one
            print("Giving default names to each dimension")
            self.names = dict((i, str(i)) for i in range(self.dims))

    # ------------------------------------------------------------------
    # Cached outputs of WESTPA tools (pdist.h5, assign.h5, ...)
    #
    # Every analysis runs from the analysis folder, with west.h5 one level up.
    # A tool's output is reused only if it was stamped as built from the same
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
        west = os.path.join("..", "west.h5")
        st = os.stat(west)
        with h5py.File(west, "r") as f:
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
            "-W", os.path.join("..", "west.h5"),
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
        command = [
            "w_assign",
            "-W", "west.h5",
            "--states-from-file", "./analysis/states.yaml",
            "-o", "./analysis/assign.h5",
        ]
        stamp = self.stamp("w_assign", states_yaml=self.file_digest("states.yaml"))
        self.run_westpa_tool("assign.h5", command, stamp, cwd="../")
        return stamp