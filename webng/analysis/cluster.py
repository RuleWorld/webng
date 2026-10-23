import pickle, h5py, os, shutil, platform
import yaml
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
        # self.system = platform.system()
        self.set_dims(self._getd(opts, "dimensions", required=False))
        self.bins = self._getd(opts, "bins", default=30, required=False)
        self.threshold = self._getd(opts, "threshold", default=90, required=True)
        self.min_samples = self._getd(opts, "min-samples", default=2, required=True)
        self.eps = self._getd(opts, "eps", default=1.5, required=True)
    
    def run(self):
        datFile = self.ensure_pdist()
        # pdist.h5 already covers exactly first_iter..last_iter, so use every row
        # (indexing it by iteration number again would double-offset the range)
        Hists = datFile["histograms"][:].mean(axis=0)

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

        print("plotting macrostates")
        figs = []
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
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
                                ax.scatter(x,y,marker=f'${l}$',s=s,c=colors[l % len(colors)],alpha=0.5)
                ax.set_xlabel(self.names[jj])
                ax.set_ylabel(self.names[ii])
                f.savefig('cluster_{}_{}.png'.format(self.names[jj],self.names[ii]))
                figs.append(f)

        print("printing macrostates")
        # Each state's coords are the midpoints of its bins, as plain floats,
        # written by a YAML library. (Formatting NumPy numbers as text gives
        # "np.float64(0.5)" under NumPy 2, which YAML reads back as a string,
        # so w_assign couldn't use the file.)
        midpoints = [datFile['midpoints_{}'.format(d)][:] for d in range(final_cluster_grid.ndim)]
        states = []
        for label in range(int(final_cluster_grid.max()) + 1):
            coords = [[float(midpoints[d][i]) for d, i in enumerate(index)]
                      for index in zip(*np.where(final_cluster_grid == label))]
            states.append({"label": "state{}".format(label), "coords": coords})
        with open("states.yaml", "w") as file:
            yaml.safe_dump({"states": states}, file, sort_keys=False, default_flow_style=None)
        
        os.chdir(self.curr_path)
        return figs