import sys, os


class weAnalysis:
    """
    Base class for all analysis classes.
    """

    def __init__(self, opts):
        # get this in so we can import system.py if need be
        sys.path.append(opts["sim_name"])
        # keep opts around
        self.opts = opts
        # Set work path
        if self.opts["work-path"] is None:
            self.work_path = os.path.join(self.opts["sim_name"], "analysis")
        # we want to go there
        if not os.path.isdir(self.work_path):
            os.mkdir(self.work_path)
        # assert os.path.isdir(self.work_path), "Work path: {} doesn't exist".format(self.work_path)
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
