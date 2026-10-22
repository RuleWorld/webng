import pickle, os, sys, platform
import numpy as np
import matplotlib.pyplot as plt
import h5py
import subprocess as sbpc
import yaml
from yaml import Loader
import shutil
# import networkx as nx
from webng.analysis.analysis import weAnalysis

# Hacky way to ignore warnings, in particular pyemma insists on Python3
import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(precision=2)


class weNetwork(weAnalysis):
    def __init__(self, opts):
        # get our parent initialization setup
        super().__init__(opts)
        # self.system = platform.system()
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        self.h5file_path = "west.h5"
        self.h5file = h5py.File(os.path.join("..",self.h5file_path), "r")
        # iterations
        self.first_iter = self._getd(opts, "first-iter", default=None, required=False)
        self.last_iter = self._getd(opts, "last-iter", default=None, required=False)
        self.first_iter, self.last_iter = self.set_iter_range(
            self.first_iter, self.last_iter
        )
        self.tau = self._getd(opts, "tau", default=10, required=True)
        self.step_iter = self._getd(opts, "step-iter", default=10, required=False)

        # check that clustering analysis was performed and there are macrostates to measure rates with
        if not os.path.isfile("states.yaml"):
            raise FileNotFoundError("states.yaml does not exist. Please run clustering analysis or assign macrostates")
        else:
            with open("states.yaml", "r") as file:
                states = yaml.load(file, Loader=Loader)
                self.state_labels = [state["label"] for state in states["states"]]

    def set_iter_range(self, first_iter, last_iter):
        if first_iter is None:
            first_iter = 1
        if last_iter is None:
            last_iter = self.h5file.attrs["west_current_iteration"] - 1

        return first_iter, last_iter

    def run(self):
        assign_stamp = self.ensure_assign()

        command = [
            "w_direct",
            "all",
            "-W", "{}".format(self.h5file_path),
            "-a", "./analysis/assign.h5",
            "--first-iter", "{}".format(self.first_iter),
            "--last-iter", "{}".format(self.last_iter),
            "--step-iter", "{}".format(self.step_iter),
            "-e", "cumulative",
        ]
        stamp = self.stamp("w_direct", assign=assign_stamp,
                           first_iter=int(self.first_iter),
                           last_iter=int(self.last_iter), step_iter=self.step_iter)
        # w_direct writes direct.h5 into the sim folder; it's moved here after
        self.run_westpa_tool("direct.h5", command, stamp, cwd="../",
                             produced_at=os.path.join("..", "direct.h5"),
                             log_path="direct_output.txt")

        dirFile = h5py.File("direct.h5", "r")
        rate_evolution = dirFile["rate_evolution"][:]
        iter_range = np.linspace(self.first_iter,self.last_iter,len(rate_evolution))

        figs = []

        for start_ind in range(len(self.state_labels)):
            for end_ind in range(len(self.state_labels)):
                if start_ind != end_ind:
                    start_state_label = self.state_labels[start_ind]
                    end_state_label = self.state_labels[end_ind]
                    means = [time[start_ind][end_ind][2] for time in rate_evolution]
                    ci_down = [time[start_ind][end_ind][3] for time in rate_evolution]
                    ci_up = [time[start_ind][end_ind][4] for time in rate_evolution]
                    f, ax = plt.subplots(figsize=(8,6))
                    ax.plot(iter_range,means,c='b')
                    ax.fill_between(iter_range,ci_down,ci_up,color='b',alpha=0.15)
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Transition Rate $\\tau^{-1}$')
                    ax.set_title(f'{start_state_label}$\\rightarrow${end_state_label}')
                    f.savefig("rate_{}_to_{}".format(start_state_label,end_state_label))
                    figs.append(f)
        os.chdir(self.curr_path)
        return figs