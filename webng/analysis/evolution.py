import os, h5py
import subprocess as sbpc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import itertools as itt
import webng.analysis.utils as utils
from webng.analysis.analysis import weAnalysis

# Hacky way to disable warnings so we can focus on important stuff
import warnings

warnings.filterwarnings("ignore")

# TODO: Separate out the pdisting, use native code
# TODO: Hook into native code to decouple some parts like # of dimensions etc.
# we need the system.py anyway, let's use native code to read CFG file
class weEvolution(weAnalysis):
    """
    Class for the evolution analysis.

    This tool creates N plots where N is the number of observables (unless overridden by
    the `dimensions` option). Each plot contains the evolution of the 1D probability distirubion
    over WE iterations.

    This tool uses `w_pdist` WESTPA tool to calculate probabilty distributions hence
    it needs `w_pdist` to be accessible directly from the commandline.
    """

    def __init__(self, opts):
        super().__init__(opts)
        # Once the arguments are parsed, do a few prep steps, opening h5file
        self.h5file_path = os.path.join(opts["sim_name"], "west.h5")
        self.h5file = h5py.File(self.h5file_path, "r")
        # Set the dimensionality
        self.set_dims(self._getd(opts, "dimensions", required=False))
        # Plotting energies or not?
        self.do_energy = self._getd(opts, "plot-energy", required=False)
        self.first_iter = self._getd(opts, "first-iter", default=None, required=False)
        self.last_iter = self._getd(opts, "last-iter", default=None, required=False)
        self.first_iter, self.last_iter = self.set_iter_range(
            self.first_iter, self.last_iter
        )
        # output name
        self.outname = self._getd(
            opts, "output", default="evolution.png", required=False
        )
        # averaging window
        self.avg_window = self._getd(opts, "avg_window", default=10, required=False)
        # color bar
        self.color_bar = self._getd(opts, "color_bar", default=False, required=False)
        self.bins = self._getd(opts, "bins", default=30, required=False)

    def set_dims(self, dims=None):
        if dims is None:
            dims = self.h5file["iterations/iter_{:08d}".format(1)]["pcoord"].shape[2]
        self.dims = dims
        # return the dimensionality if we need to
        return self.dims

    def set_names(self, names):
        if names is not None:
            self.names = dict(zip(range(len(names)), names))
        else:
            # We know the dimensionality, can assume a
            # naming scheme if we don't have one
            print("Giving default names to each dimension")
            self.names = dict((i, str(i)) for i in range(self.dims))

    def set_iter_range(self, first_iter, last_iter):
        if first_iter is None:
            first_iter = 1
        if last_iter is None:
            last_iter = self.h5file.attrs["west_current_iteration"] - 1

        return first_iter, last_iter

    def setup_figure(self):
        # Setup the figure and names for each dimension
        # plt.figure(figsize=(20,20))
        plt.figure(figsize=(1.5, 3.0))
        f, axarr = plt.subplots(1, self.dims)
        f.subplots_adjust(
            hspace=1.2, wspace=0.2, bottom=0.1, left=0.06, top=0.98, right=0.98
        )
        f.supxlabel("WE Iterations")
        axarr = np.atleast_2d(axarr)
        return f, axarr

    def save_fig(self):
        # setup our output filename
        if self.outname is not None:
            outname = self.outname
        else:
            outname = "all_{:05d}.png".format(self.last_iter)

        # save the figure
        print("Saving figure to {}".format(outname))
        plt.savefig(outname, dpi=600)
        plt.close()
        return

    def run(self, ext=None):
        if not os.path.isfile("pdist.h5"):
            print("pdist.h5 does not exist. Running w_pdist")
            proc = sbpc.Popen(
                    [
                        "w_pdist",
                        "-W",
                        "{}".format(self.h5file_path),
                        "--first-iter",
                        "{}".format(self.first_iter),
                        "--last-iter",
                        "{}".format(self.last_iter),
                        "-o",
                        "pdist.h5",
                        "-b",
                        "{}".format(self.bins)
                    ]
                )
            proc.wait()

        datFile = h5py.File("pdist.h5", "r")

        if "plot-opts" in self.opts:
            plot_opts = self.opts["plot-opts"]
            name_fsize = self._getd(plot_opts, "name-font-size", default=6)

        f, axarr = self.setup_figure()

        # Loop over every dimension vs every other dimension
        for i in range(self.dims):
            print("Plotting dimension {}".format(i + 1))

            # It's too messy to plot the spines and ticks for large dimensions
            for kw in ["top", "right"]:
                axarr[0, i].spines[kw].set_visible(False)
            axarr[0, i].tick_params(left=False, bottom=False)

            # Set the names
            axarr[0, i].set_ylabel(self.names[i], fontsize=name_fsize)

            # First pull a file that contains the dimension
            Hists = datFile["histograms"][:]
            axes_to_average = tuple(d for d in range(1,self.dims+1) if d != i+1)
            Hists = Hists.mean(axis=axes_to_average)

            # moving_avg = []
            # for starti in range(1, self.last_iter - self.avg_window):
            #     prob = Hists[starti : starti + self.avg_window].mean(axis=0)
            #     prob = prob / prob.sum()
            #     if not self.do_energy:
            #         prob = prob / prob.max()
            #     moving_avg.append(prob)
            # Hists = np.array(moving_avg)
            if self.do_energy:
                Hists = -np.log(Hists)
                Hists = Hists - Hists.min()

            # Calculate the x values, normalize s.t. it spans 0-1
            x_bins = datFile["midpoints_{}".format(i)][...]

            # Plot on the correct ax, set x limit
            cmap = mpl.cm.magma_r
            cmap.set_bad(color="white")
            cmap.set_over(color="white")
            cmap.set_under(color="white")
            
            pcolormesh = axarr[0, i].pcolormesh(
                range(self.first_iter-1,self.last_iter),
                x_bins,
                Hists.T,
                cmap=cmap,
                vmin=1e-60,
            )

            if self.color_bar:
                f.colorbar(
                    pcolormesh,
                    ax=axarr[0, i]
                )

        plt.tight_layout()
        self.save_fig()
        return
