.. _config:

#####################
Configuration Options
#####################

Simulation setup options
========================

The section that's relevant for the simulation setup should look something like this:

.. code-block:: yaml
   :linenos:

   binning_options:
      block_size: 10 # Number of trajectories to be processed in blocks
      center_freq: 1 # How frequently do we add new Voronoi centers?
      max_centers: 300 # Maximum number of Voronoi centers to be added
      traj_per_bin: 100 # Number of trajectories per Voronoi center
   path_options: # this entire section should be automatically set by the tool
      WESTPA_path: /home/USER/westpa
      bng_path: /home/USER/apps/anaconda3/lib/python3.7/site-packages/bionetgen/bng-linux
      bngl_file: /home/USER/webng/testing/test.bngl
      sim_name: /home/USER/webng/testing/test # you can adjust sim folder here
   propagator_options:
      pcoords: # These should match observables in your model
      - Atot
      - Btot
      propagator_type: libRoadRunner # this is the suggested propagator
   sampling_options:
      dimensions: 2 # Dimensionality of the WESTPA progress coordinates
      max_iter: 10 # Maximum number of WE iterations
      pcoord_length: 10 # Number of data points per WE iteration
      tau: 100 # Resampling frequency

you can change various aspects of the simulation setup in this file. 
Let's look at each block separately. 

.. _binning:

Binning
-------

.. code-block:: yaml
   :linenos:

   binning_options:
      block_size: 10 # Number of trajectories to be processed in blocks
      center_freq: 1 # How frequently do we add new Voronoi centers?
      max_centers: 300 # Maximum number of Voronoi centers to be added
      traj_per_bin: 100 # Number of trajectories per Voronoi center

:code:`block_size` refers to how many trajectories will be ran at a time. This is important for
multicore runs, try to keep the blocksize an integer multiple of the number of cores you have. 
:code:`center_freq` refers to how frequently voronoi bins will be placed, in units of WE 
iterations. :code:`max_centers` is the maximum number of voronoi centers that will be placed.
Finally, :code:`traj_per_bin` is the number of trajectories in each voronoi center.

.. _mabl:

MABL
^^^^

With :code:`style: mabl` all walkers share one bin, and the driver
(:code:`MABL_driver.py`, copied from :code:`webng/assets` with its settings block
filled in) resamples by score. MABL always recycles: the :code:`recycling` block's
:code:`recycle_region` is the target. Lists have one entry per progress coordinate,
in :code:`pcoords` order.

.. code-block:: yaml
   :linenos:

   binning_options:
      style: mabl
      n_walkers: 1000          # walkers, constant through the run
      n_split: 100             # walkers split per iteration; as many pairwise merges follow
      weight_cap_factor: 10.0  # split any walker heavier than this x the average weight
      start: [100, 300, 3250, 100]
      target: [1000, 300, 100, 3350]
      coord_weights: [0, 0, 1, 1]
      log_scale: [false, false, false, true]
      merge_progress_floor: 0.01
      local_merge: false
      merge_pool: 4
      merge_log: false

Each iteration, after recycling, MABL computes every walker's progress
:math:`S = \sum_d w_d\, p_d`, where :math:`w_d` is :code:`coord_weights` and
:math:`p_d \in [0, 1]` measures how far coordinate :math:`d` has come from
:code:`start` toward :code:`target`. With :code:`log_scale` false, :math:`p_d` is
linear in the count (:math:`1 - |x - \mathrm{target}| / |\mathrm{start} - \mathrm{target}|`).
With :code:`log_scale` true, it is the fold change
(:math:`\log(x/\mathrm{start}) / \log(\mathrm{target}/\mathrm{start})`, which needs start
and target > 0). The log scale is sensitive at low counts, so it suits a coordinate
whose transition happens as it rises from a low start. Whether it helps is case by
case (default false; a single true/false applies to every coordinate).

The :code:`n_split` walkers with the highest :math:`S / -\log W` are split in two.
Then as many pairwise merges remove the added walkers. Candidates are the walkers
with the highest merge score :math:`1 / (W \max(S, \mathrm{floor}))`, light and
unpromising, where the floor is :code:`merge_progress_floor`. Without the floor,
walkers with zero progress would be merged first however heavy they are.
Walkers created by a split in the same iteration are never merge candidates: a
split's two children sit at the same point with equal weights, so they would pair
up and undo the split.
By default candidates are paired by merge-score rank.
With :code:`local_merge: true`, each candidate is instead paired with its nearest
neighbour (Euclidean distance in the progress coordinates). Partners come from the
top :code:`merge_pool` :math:`\times` (merges per iteration) walkers by merge score,
so a larger pool finds closer partners from lower in the ranking, and 2 merges
exactly the top-ranked walkers. Merges are always
pairwise, so no merge makes a walker heavier than the two it came from, and the
survivor is chosen by weight. :code:`merge_log: true` writes every merge to
:code:`merge_dists.tsv` in the simulation folder (for debugging).

By default all walkers compete for the :code:`n_split` splits in one ranking. When a
transition has two legs, walkers on the first leg (e.g. still in the start basin)
may never win a split against walkers further along. :code:`split_bands` (ascending
thresholds on :math:`S`) and :code:`split_fractions` (one share of :code:`n_split`
per band, summing to 1) give each band its own budget, for example
:code:`split_bands: [0.5]` with :code:`split_fractions: [0.3, 0.7]` reserves 30% of
the splits for walkers with :math:`S < 0.5`. Within a band walkers are ranked as
usual, and a band with too few walkers passes its unused splits on.

.. _path_options:

Path Options
------------

.. code-block:: yaml
   :linenos:

   path_options: # this entire section should be automatically set by the tool
      WESTPA_path: /home/USER/westpa
      bng_path: /home/USER/apps/anaconda3/lib/python3.7/site-packages/bionetgen/bng-linux
      bngl_file: /home/USER/webng/testing/test.bngl
      sim_name: /home/USER/webng/testing/test # you can adjust sim folder here

Most of these option should be set automatically if WESTPA and BNG are both python importable. 
:code:`WESTPA_path` is the path to WESTPA to be used, :code:`bng_path` is the path where BNG2.pl 
lives. :code:`bngl_file` is the bngl model and :code:`sim_name` is the folder that will be used 
for the WESTPA setup. 

.. _propagator_options:

Propagator Options
------------------

.. code-block:: yaml
   :linenos:

   propagator_options:
      pcoords: # These should match observables in your model
      - Atot
      - Btot
      propagator_type: libRoadRunner # this is the suggested propagator

:code:`pcoords` is the list progress coordinates to be used for WESTPA and should match the 
observables in your BNGL model. :code:`propagator_type` is the type of propagator to be used. If 
available, use libRoadRunner since it's currently significantly more efficient for WESTPA runs. 
If not, you can select "executable" propagator which uses BNG2.pl in combination with bash scripts
for each walker. 

.. _propagator_options:

Sampling Options
------------------

.. code-block:: yaml
   :linenos:

   sampling_options:
      dimensions: 2 # Dimensionality of the WESTPA progress coordinates
      max_iter: 10 # Maximum number of WE iterations
      pcoord_length: 10 # Number of data points per WE iteration
      tau: 100 # Resampling frequency

:code:`dimensions` is the number of dimensions to be used for WESTPA progress coordinates and 
should match the number of BNGL observables you are using. :code:`max_iter` is the maximum number
of WE iterations to be ran (this can be changed later from within the setup). :code:`pcoord_length`
is the number of data points each walker will return. :code:`tau` is the length of each BNGL
simulation/walker. 

Analysis options
================

When you first create a setup configuration file like :code:`mysim.yaml`, you will see
an analysis section like this

.. code-block:: yaml
   :linenos:

   analyses:
      enabled: false
      work-path: /home/USER/webng/testing/test/analysis # the folder to run the analysis under
      average:
         dimensions: null # you can limit the tool to the first N dimensions
         enabled: false # this needs to be set to true to run the analysis 
         first-iter: null # first iteration to start the averaging
         last-iter: null # first iteration to end the averaging
         mapper-iter: null # the iteration to pull the voronoi bin mapper from, last iteration by default
         normalize: false # normalizes the distributions
         output: average.png # output file name 
         plot-energy: false # plots -ln of probabilies
         plot-opts: # various plotting options like font sizes and line width
            name-font-size: 12
            voronoi-col: 0.75
            voronoi-lw: 1
         plot-voronoi: false # true if you want to plot voronoi centers
         smoothing: 0.5 # the amount of smoothing to apply
      evolution:
         avg_window: null # number of iterations to average for each point in the plot
         dimensions: null # you can limit the tool to the first N dimensions
         enabled: false # this needs to be set to true to run the analysis
         normalize: false # normalizes the distributions
         output: evolution.png # output file name 
         plot-energy: false # plots -ln of probabilies
         plot-opts: # various plotting options like font sizes and line width
            name-font-size: 12

Let's take a look at individual sections. 

.. code-block:: yaml
   :linenos:

   analyses:
      enabled: false
      work-path: /home/USER/webng/testing/test/analysis # the folder to run the analysis under

This is upper level analysis block and has a single option called :code:`enabled`. If set to false,
none of the analyses will run. Each analysis subsection will have the same :code:`enabled` option
to set if that particular analysis will be ran or not. :code:`work-path` is the folder where all 
analysis will be ran. 

.. _average:

Average
-------

.. code-block:: yaml
   :linenos:

   average:
      dimensions: null # you can limit the tool to the first N dimensions
      enabled: false # this needs to be set to true to run the analysis 
      first-iter: null # first iteration to start the averaging
      last-iter: null # first iteration to end the averaging
      mapper-iter: null # the iteration to pull the voronoi bin mapper from, last iteration by default
      normalize: false # normalizes the distributions
      output: average.png # output file name 
      plot-energy: false # plots -ln of probabilies
      plot-opts: # various plotting options like font sizes and line width
         name-font-size: 12
         voronoi-col: 0.75
         voronoi-lw: 1
      plot-voronoi: false # true if you want to plot voronoi centers
      smoothing: 0.5 # the amount of smoothing to apply

This is the block for :ref:`Average analysis`. :code:`dimensions` is normally set to null which
makes the tool plot all dimensions. If this is set to :code:`N` the tool will plot the first 
:code:`N` dimensions. :code:`first-iter` and :code:`last-iter` are the iterations to start and 
stop the averaging. :code:`mapper-iter` is the iteration to pull the voronoi mapper from, if you 
don't want the mapper from the final WE iteration. :code:`normalize` can be used to enable 
normalization of probability distributions before plotting. :code:`output` is the file name 
for the output and this can be set to a :code:`png` or :code:`pdf` file. :code:`plot-energy` 
takes the :code:`-ln` of the probabilities before plotting. :code:`plot-voronoi` controls 
if the voronoi centers are plotted on top of the probability distributions. :code:`smoothing`
can be changed to reduce or increase the gaussian smoothing used for probability distributions. 
:code:`plot-opts` contain some options for plotting. :code:`name-front-size` is the font-size used
in plotting. :code:`voronoi-col` is the color to be used for voronoi bins and :code:`voronoi-lw` 
is the line width for the same lines. 

.. _evolution:

Evolution
---------

.. code-block:: yaml
   :linenos:

   evolution:
      avg_window: 1 # number of iterations to average for each point in the plot
      dimensions: null # you can limit the tool to the first N dimensions
      enabled: false # this needs to be set to true to run the analysis
      normalize: false # normalizes the distributions
      output: evolution.png # output file name 
      plot-energy: false # plots -ln of probabilies
      plot-opts: # various plotting options like font sizes and line width
         name-font-size: 12

This is the block for :ref:`Evolution analysis`. :code:`avg_window` the number of iterations to
average over for every data point. :code:`dimensions` is normally set to null which
makes the tool plot all dimensions. If this is set to :code:`N` the tool will plot the first 
:code:`N` dimensions. :code:`normalize` can be used to enable normalization of probability 
distributions before plotting. :code:`output` is the file name for the output and this can be 
set to a :code:`png` or :code:`pdf` file. :code:`plot-opts` contain some options for plotting. 
:code:`name-front-size` is the font-size used in plotting. 

.. _cluster:

Cluster
-------

.. code-block:: yaml
   :linenos:

   cluster:
      enabled: true
      method: dbscan          # dbscan (density) or gpcca (dynamics)
      probability-mass: 0.9   # dbscan
      eps: null               # dbscan
      min-samples: 2          # dbscan
      log-scale: false        # dbscan
      n-clusters: null        # gpcca
      max-clusters: 10        # gpcca

This is the block for :ref:`Cluster analysis`, which finds macrostates and writes them to
:code:`states.yaml` (each state a list of coordinates; every WESTPA bin containing one of them
belongs to that state). :code:`method` picks how:

* :code:`dbscan` keeps the most probable WESTPA bins until they hold :code:`probability-mass` of
  the probability (each bin's average probability over the iterations), then groups the kept bins
  that lie close together with DBSCAN. Distances are measured with every progress coordinate
  scaled to [0, 1] over the range the run covered (after :code:`log10(x + 1)` with
  :code:`log-scale: true`, for coordinates spanning orders of magnitude), so :code:`eps` is a
  fraction of that range; :code:`null` chooses twice the median distance from a kept bin to its
  nearest kept neighbour. :code:`min-samples` counts bins, the bin itself included. Bins left out,
  or that DBSCAN calls noise, belong to no state, and states holding little probability are cut
  first (that is what :code:`gpcca` is for). Also writes :code:`cluster_dbscan.txt`.
* :code:`gpcca` runs GPCCA+ (`pyGPCCA <https://github.com/msmdev/pyGPCCA>`_) on the run's
  bin-to-bin transition matrix at lag tau, made reversible by symmetrizing the stationary flux so
  each state keeps its probability. Every bin the walkers move back and forth through gets a
  state, including states that hold little probability. :code:`n-clusters` is the number of
  states, or :code:`null` to let GPCCA+ choose between 2 and :code:`max-clusters` by crispness.
  Also writes :code:`cluster_gpcca.txt` (each state's stationary probability and the
  state-to-state transition probabilities).

Both use the run's own bins (a :code:`w_assign` of the run, :code:`assign_bins.h5`), or the file
named by :code:`analyses: assign-h5`, and list one coordinate per bin in :code:`states.yaml`: a
progress coordinate the run actually had in that bin.

.. _network:

Network
-------

.. code-block:: yaml
   :linenos:

   network:
      enabled: true
      step-iter: 10
      matrix: true

This is the block for :ref:`Network generation`: rates and mean first passage times (MFPTs)
between the states of :code:`states.yaml`, from either cluster method or written by hand. It runs
:code:`w_assign` with those states, then estimates them two ways. :code:`w_direct` labels each
walker with the state it visited last and divides the flux of A-labelled walkers into B by the
A-labelled probability (rates per tau, plotted as they evolve). With :code:`matrix: true`,
the bin-to-bin transition matrix at lag tau (each segment's weight counted from the bin it
started in to the bin it ended in) is lumped into the
states and used for MFPTs (the expected time to first reach B, averaged over A's bins weighted
by stationary probability), plotted in :code:`mfpt_evolution.png`. :code:`step-iter` is how often
the estimates are evaluated. Both are summarized in :code:`network.txt`, with MFPTs in real time
(iterations times tau).
