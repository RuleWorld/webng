import os, h5py, sys, platform
import scipy.ndimage
import subprocess as sbpc
import yaml
from yaml import Loader
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
class weAverage(weAnalysis):
    """
    Class for the averaging analysis.

    This tool creates a N by N matrix-like plot where N is the number of observables
    in the BNGL tool (unless overridden by the `dimensions` option). The diagonal
    will contain 1D probability distributions and off diagonals will contain 2D probability
    heatmaps of each dimension vs each other dimension.

    This tool uses `w_pdist` WESTPA tool to calculate probabilty distributions hence
    it needs `w_pdist` to be accessible directly from the commandline.
    """

    def __init__(self, opts):
        super().__init__(opts)
        # Once the arguments are parsed, do a few prep steps, opening h5file
        # self.system = platform.system()
        self.h5file_path = os.path.join("..", "west.h5")
        self.h5file = h5py.File(self.h5file_path, "r")
        # Set the dimensionality
        self.set_dims(self._getd(opts, "dimensions", required=False))
        # Plot the bins?
        self.plot_boundaries = self._getd(opts, "plot-boundaries", default=False, required=False)
        # Plotting energies or not?
        self.do_energy = self._getd(opts, "plot-energy", default=False, required=False)
        # iterations
        self.first_iter = self._getd(opts, "first-iter", default=None, required=False)
        self.last_iter = self._getd(opts, "last-iter", default=None, required=False)
        self.first_iter, self.last_iter = self.set_iter_range(
            self.first_iter, self.last_iter
        )
        # output name
        self.outname = self._getd(opts, "output", default="average.png", required=False)
        # data smoothing
        self.data_smoothing_level = self._getd(opts, "smoothing", default=0.5, required=False)
        # data normalization to min/max
        self.normalize = self._getd(opts, "normalize", default=False, required=False)
        # get color bar option
        self.color_bar = self._getd(opts, "color_bar", default=True, required=False)
        # get analysis bins
        self.bins = self._getd(opts, "bins", default=30, required=False)

        # handle plotting bin boundaries if needed
        self.style = None
        self.boundaries = None
        self.mapper = None
        if self.plot_boundaries:
            try:
                from system import System
                sys_obj = System()
                sys_obj.initialize()
                # Detect style from the mapper type
                mapper_type = type(sys_obj.bin_mapper).__name__
                if mapper_type == "RectilinearBinMapper":
                    self.style = "regular"
                    self.boundaries = sys_obj.bin_mapper.boundaries
                elif mapper_type == "SingleBinMapper":
                    self.style = "mabl"
                    # plot_boundaries is silently ignored for MABL
                else:
                    # Assume adaptive (VoronoiBinMapper or similar)
                    self.style = "adaptive"
                    self.mapper = self.get_mapper(
                        self._getd(opts, "mapper-iter", default=None, required=False)
                    )
            except Exception as e:
                print("Warning: could not determine binning style for boundary "
                    "plotting: {}".format(e))

    def get_mapper(self, mapper_iter):
        # Gotta fix this behavior
        if mapper_iter is None:
            mapper_iter = self.h5file.attrs["west_current_iteration"] - 1
        # Load in mapper from the iteration given/found
        print(
            "Loading file {}, mapper from iteration {}".format(
                self.h5file_path, mapper_iter
            )
        )
        # We have to rewrite this behavior to always have A mapper from somewhere
        # and warn the user appropriately, atm this is very shaky
        try:
            self.mapper = utils.load_mapper(self.h5file, mapper_iter)
        except:
            self.mapper = utils.load_mapper(self.h5file, mapper_iter - 1)

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
        plt.figure(figsize=(1.5, 1.5))
        f, axarr = plt.subplots(self.dims, self.dims)
        axarr = np.atleast_2d(axarr)
        f.subplots_adjust(
            hspace=0.4, wspace=0.4, bottom=0.05, left=0.05, top=0.98, right=0.9
        )
        return f, axarr

    def save_fig(self):
        # setup our output filename
        if self.outname is not None:
            outname = self.outname
        else:
            outname = "all_{:05d}_{:05d}.png".format(self.first_iter, self.last_iter)

        # save the figure
        print("Saving figure to {}".format(outname))
        plt.savefig(outname, dpi=600)
        return

    def run(self, ext=None):
        if not os.path.isfile("pdist.h5"):
            print("pdist.h5 does not exist. Running w_pdist")
            command = [
                "w_pdist",
                "-W", "{}".format(self.h5file_path),
                "--first-iter", "{}".format(self.first_iter),
                "--last-iter", "{}".format(self.last_iter),
                "-o", "pdist.h5",
                "-b", "{}".format(self.bins),
            ]
            proc = sbpc.Popen(command)
            proc.wait()
        datFile = h5py.File("pdist.h5", "r")

        # Plot options
        name_fsize   = 6
        line_width   = 0.15
        line_color   = "0.75"
        if "plot-opts" in self.opts:
            plot_opts  = self.opts["plot-opts"]
            name_fsize = self._getd(plot_opts, "name-font-size", default=6)
            line_width = self._getd(plot_opts, "line_width", default=0.15)
            line_color = str(self._getd(plot_opts, "line-col", default=0.75))

        f, axarr = self.setup_figure()

        for jj in range(self.dims):
            for ii in range(jj, self.dims):
                Hists = datFile["histograms"][:].mean(axis=0)

                print("Plotting {} vs {}".format(ii + 1, jj + 1))
                fi, fj = ii + 1, jj + 1

                for kw in ["top", "right"]:
                    axarr[ii, jj].spines[kw].set_visible(False)
                axarr[ii, jj].tick_params(left=False, bottom=False)

                if ii != jj:
                    for kw in ["top", "bottom", "left", "right"]:
                        axarr[jj, ii].spines[kw].set_visible(False)
                    axarr[jj, ii].set_xticks([])
                    axarr[jj, ii].set_yticks([])

                if fi == self.dims:
                    axarr[ii, jj].set_xlabel(self.names[jj], fontsize=name_fsize)
                if fj == 1:
                    axarr[ii, jj].set_ylabel(self.names[ii], fontsize=name_fsize)

                if fi == fj:
                    # ---- 1D diagonal plot ----
                    if self.normalize:
                        axarr[ii, jj].set(adjustable="box", aspect="equal")
                    axes_to_average = tuple(d for d in range(self.dims) if d != ii)
                    hist1d = Hists.mean(axis=axes_to_average)
                    hist1d = hist1d / hist1d.flatten().sum()
                    if self.do_energy:
                        hist1d = -np.log(hist1d)
                    x_mids = datFile["midpoints_{}".format(ii)][...]
                    if self.normalize:
                        x_mids = x_mids / x_mids.max()
                        axarr[ii, jj].set_xlim(0.0, 1.0)
                        axarr[ii, jj].set_ylim(0.0, 1.0)
                    axarr[ii, jj].plot(x_mids, hist1d)

                    # -- Regular bin boundaries on 1D plot --
                    if self.style == "regular" and self.boundaries is not None:
                        for bval in self.boundaries[ii]:
                            axarr[ii, jj].axvline(
                                x=bval, color=line_color, lw=line_width
                            )

                else:
                    # ---- 2D heatmap ----
                    if self.normalize:
                        axarr[ii, jj].set(adjustable="box", aspect="equal")
                    axes_to_average = tuple(
                        d for d in range(self.dims) if d != ii and d != jj
                    )
                    hist2d = Hists.mean(axis=axes_to_average)
                    hist2d = hist2d / hist2d.sum()
                    hist2d[np.isnan(hist2d)] = np.nanmax(hist2d)
                    if self.do_energy:
                        hist2d = -np.log(hist2d)
                    if self.data_smoothing_level is not None:
                        hist2d = scipy.ndimage.filters.gaussian_filter(
                            hist2d, self.data_smoothing_level
                        )
                    e_dist = hist2d.T

                    x_bins = datFile["midpoints_{}".format(ii)][...]
                    y_bins = datFile["midpoints_{}".format(jj)][...]
                    if self.normalize:
                        x_max = x_bins.max()
                        y_max = y_bins.max()
                        if x_max != 0:
                            x_bins = x_bins / x_max
                        if y_max != 0:
                            y_bins = y_bins / y_max
                        axarr[ii, jj].set_xlim(0.0, 1.0)
                        axarr[ii, jj].set_ylim(0.0, 1.0)

                    cmap = mpl.cm.magma_r
                    cmap.set_bad(color="white")
                    cmap.set_over(color="white")
                    cmap.set_under(color="white")

                    pcolormesh = axarr[ii, jj].pcolormesh(
                        y_bins, x_bins, e_dist, cmap=cmap, vmin=1e-10
                    )

                    if self.color_bar:
                        cbar = f.colorbar(pcolormesh, ax=axarr[ii, jj])
                        cbar.ax.tick_params(labelsize=name_fsize)

                    # -- Regular bin boundaries on 2D plot --
                    if self.boundaries is not None:
                        # jj is the x-axis dimension, ii is the y-axis dimension
                        for bval in self.boundaries[jj]:
                            axarr[ii, jj].axvline(
                                x=bval, color=line_color, lw=line_width
                            )
                        for bval in self.boundaries[ii]:
                            axarr[ii, jj].axhline(
                                y=bval, color=line_color, lw=line_width
                            )

                    # -- Voronoi bins on 2D plot (adaptive) --
                    if self.plot_boundaries:
                        try:
                            X = self.mapper.centers[:, ii]
                            Y = self.mapper.centers[:, jj]
                            if self.normalize:
                                X = X / y_max
                                Y = Y / x_max
                            if not ((X == 0).all() or (Y == 0).all()):
                                axarr[ii, jj].scatter(Y, X, s=0.1)
                                segments = utils.voronoi(Y, X)
                                lines = mpl.collections.LineCollection(
                                    segments, color=line_color, lw=line_width
                                )
                                axarr[ii, jj].add_collection(lines)
                                axarr[ii, jj].ticklabel_format(style="sci")
                        except Exception:
                            pass

        f.tight_layout(rect=[0.1, 0.1, 1, 1])
        f.subplots_adjust(hspace=0.3, wspace=0.4, top=0.98, left=0.1, bottom=0.1)
        for ax in axarr.flat:
            ax.tick_params(axis='both', which='major', labelsize=name_fsize)

        self.save_fig()
        os.chdir(self.curr_path)
        return f, axarr