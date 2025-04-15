import pickle, h5py, os, shutil
from sys import stdout
import numpy as np
import matplotlib.pyplot as plt
# import pygpcca as pgp
import subprocess as sbpc
# from scipy.sparse import coo_matrix
from sklearn.cluster import DBSCAN
from webng.analysis.analysis import weAnalysis

# Hacky way to ignore warnings, in particular pyemma insists on Python3
import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(precision=2)


class weCluster(weAnalysis):
    def __init__(self, opts):
        super().__init__(opts)
        # Parse and set the arguments
        # get west.h5 path
        self.h5file_path = os.path.join("..", "west.h5")
        self.h5file = h5py.File(self.h5file_path, "r")
        self.set_dims(self._getd(opts, "dimensions", required=False))
        # iterations
        self.first_iter = self._getd(opts, "first-iter", default=None, required=False)
        self.last_iter = self._getd(opts, "last-iter", default=None, required=False)
        self.first_iter, self.last_iter = self.set_iter_range(
            self.first_iter, self.last_iter
        )
        self.bins = self._getd(opts, "bins", default=30, required=False)
        self.threshold = self._getd(opts, "density-threshold", default=90, required=True)
        self.min_samples = self._getd(opts, "min-samples", default=2, required=True)
        self.eps = self._getd(opts, "eps", default=1.5, required=True)

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

    def run(self):
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

        # Extract pdist file and average out the iterations
        datFile = h5py.File("pdist.h5", "r")
        Hists = datFile["histograms"][self.first_iter:self.last_iter].mean(axis=0)

        # Figure out which bins are contained in high density regions
        density_threshold = np.percentile(Hists, self.threshold)
        high_density_bins = Hists > density_threshold
        dense_coords = np.column_stack(np.where(high_density_bins))

        # Run DBSCAN to cluster bins
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(dense_coords)
        cluster_labels = dbscan.labels_
        final_cluster_grid = np.full(Hists.shape,-1)
        for coord, label in zip(dense_coords, cluster_labels):
            final_cluster_grid[tuple(coord)] = label

        yaml_texts = [""] * np.max(final_cluster_grid+1)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        figs = []

        for jj in range(self.dims):
            for ii in range(jj+1,self.dims):
                label_grid = [[set() for _ in range(self.bins)] for _ in range(self.bins)]
                for index in np.ndindex(final_cluster_grid.shape):
                    x_ind = index[jj]
                    y_ind = index[ii]
                    label = final_cluster_grid[index]
                    if label != -1:
                        label_grid[x_ind][y_ind].add(label)
                label_grid = [[sorted(list(cell)) for cell in row] for row in label_grid]

                x_mid = datFile['midpoints_{}'.format(jj)][:]
                y_mid = datFile['midpoints_{}'.format(ii)][:]

                # Create a scatterplot displaying macrostates
                f, ax = plt.subplots(figsize=(8,6))
                for ix,x in enumerate(x_mid):
                    for iy,y in enumerate(y_mid):
                        label = label_grid[ix][iy]
                        if label == []:
                            s = 3
                            ax.scatter(x,y,marker='o',s=s,c='k')
                        else:
                            s = 80
                            for l in label:
                                ax.scatter(x,y,marker=f'${l}$',s=s,c=colors[l],alpha=0.5)
                                yaml_texts[l] += f"\n      - [{x}, {y}]"
                ax.set_xlabel(self.names[jj])
                ax.set_ylabel(self.names[ii])
                f.savefig('cluster_{}_{}.png'.format(self.names[jj],self.names[ii]))
                figs.append(f)

        final_yaml_text = "states:"
        for i in range(np.max(final_cluster_grid)+1):
            final_yaml_text += f"\n  - label: state{i}\n    coords:"
            final_yaml_text += yaml_texts[i]
        with open("states.yaml", "w") as file:
            file.write(final_yaml_text)
        os.chdir(self.curr_path)
        return figs
