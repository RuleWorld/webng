import pickle, os, sys
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
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        self.h5file_path = os.path.join("..", "west.h5")
        self.h5file = h5py.File(self.h5file_path, "r")
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
        if not os.path.isfile("assign.h5"):
            print("assign.h5 does not exist. Running w_assign")
            proc = sbpc.Popen(
                [
                    "w_assign",
                    "-W",
                    "{}".format(self.h5file_path),
                    "--states-from-file",
                    "./analysis/states.yaml",
                    "-o",
                    "./analysis/assign.h5",
                ]
            ,cwd="../")
            # OSDEPEND: Assumes Unix, ../. THIS SHOULD STILL WORK
            proc.wait()

        if not os.path.isfile("direct.h5"):
            print("direct.h5 does not exist. Running w_direct")
            with open("direct_output.txt", "w") as f:
                proc = sbpc.Popen(
                    [
                        "w_direct",
                        "all",
                        "-W",
                        "{}".format(self.h5file_path),
                        "-a",
                        "./analysis/assign.h5",
                        "--first-iter",
                        "{}".format(self.first_iter),
                        "--last-iter",
                        "{}".format(self.last_iter),
                        "--step-iter",
                        "{}".format(self.step_iter),
                        "-e",
                        "cumulative"
                    ]
                ,stdout=sbpc.PIPE, stderr=sbpc.STDOUT, text=True, cwd="../")
                # OSDEPEND: Assumes Unix, ../ THIS SHOULD STILL WORK
                proc.wait()
                for line in proc.stdout:
                    print(line, end="")
                    if not line.strip().endswith("..."):
                        f.write(line)
            shutil.move(os.path.join("..", "direct.h5"), "direct.h5")

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
