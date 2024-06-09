.. TSB-kit documentation master file, created by
   sphinx-quickstart on Sun Jun  9 18:10:29 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TSB-kit's documentation!
===================================

.. toctree::
   :maxdepth: 1
   :hidden:

   overview/index
   algorithms/index
   evaluation/index
   data-processing/index
   



Overview
--------

TSB-kit is a library of univariate time-series anomaly detection methods from the `TSB_UAD benchmark <https://github.com/TheDatumOrg/TSB-UAD>`_. Overall, TSB-kit contains 14 anomaly detection methods, and 15 evaluation measures. TSB-kit is part of a lrger project called TSB-UAD. The latter provides several time series anomaly detection datasets:

1. `Real data <https://www.thedatum.org/datasets/TSB-UAD-Public.zip>`_
2. `Synthetic <https://www.thedatum.org/datasets/TSB-UAD-Synthetic.zip>`_
3. `Artificial <https://www.thedatum.org/datasets/TSB-UAD-Artificial.zip>`_


Installation
^^^^^^^^^^^^

Quick start:


TSB-kit is published to `PyPI <https://pypi.org>`_ and you can install it using:

.. code-block:: bash

   pip install tsb-kit

.. attention::

   Currently, TSB-kit is tested only with Python 3.8.

Manual installation:

The following tools are required to install TSB-kit from source:

- git
- conda (anaconda or miniconda)


Clone this `repository <https://github.com/boniolp/tsb-kit>`_ using git and go into its root directory.

.. code-block:: bash

   git clone https://github.com/boniolp/tsb-kit.git
   cd tsb-kit/

Create and activate a conda-environment 'tsb-kit'.

.. code-block:: bash

   conda env create --file environment.yml
   conda activate tsb-kit

You can then install TSB-kit with pip.

.. code-block:: bash

   pip install tsb-kit

Usage
^^^^^

We depicts below a code snippet demonstrating how to use one anomaly detector (in this example, IForest).

.. code-block:: python

   import os
   import numpy as np
   import pandas as pd
   from tsb_kit.models.iforest import IForest
   from tsb_kit.models.feature import Window
   from tsb_kit.utils.slidingWindows import find_length
   from tsb_kit.vus.metrics import get_metrics

   df = pd.read_csv('data/benchmark/ECG/MBA_ECG805_data.out', header=None).to_numpy()
   data = df[:, 0].astype(float)
   label = df[:, 1]

   slidingWindow = find_length(data)
   X_data = Window(window = slidingWindow).convert(data).to_numpy()

   clf = IForest(n_jobs=1)
   clf.fit(X_data)
   score = clf.decision_scores_

   score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
   score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))


   results = get_metrics(score, label, metric="all", slidingWindow=slidingWindow)
   for metric in results.keys():
       print(metric, ':', results[metric])

.. code-block:: bash

   AUC_ROC : 0.9216216369841076
   AUC_PR : 0.6608577550833885
   Precision : 0.7342093339374717
   Recall : 0.4010891089108911
   F : 0.5187770129662238
   Precision_at_k : 0.4010891089108911
   Rprecision : 0.7486112853253205
   Rrecall : 0.3097733542316151
   RF : 0.438214653167952
   R_AUC_ROC : 0.989123018780308
   R_AUC_PR : 0.9435238401582703
   VUS_ROC : 0.9734357459251715
   VUS_PR : 0.8858037295594041
   Affiliation_Precision : 0.9630674176380548
   Affiliation_Recall : 0.9809813654809071