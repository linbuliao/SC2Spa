.. _api:

API
===
Import SC2Spa::

    import SC2Spa

Spatial Inference
~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: SC2Spa.tl
.. currentmodule:: SC2Spa

.. autosummary::
    :toctree: api

    tl.FineMapping
    tl.Reconstruct_scST
    tl.NRD_CT_preprocess
    tl.NRD_weight
    tl.NRD_CT
    tl.NRD_impute
    tl.Train_transfer
    tl.SaveValidation
    tl.CheckAccuracy
    tl.CrossValidation
    tl.WassersteinD
    tl.pp_Mapping
    tl.BatchPredict
    tl.Self_Mapping
    tl.Mapping

Plotting
~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: SC2Spa.pl
.. currentmodule:: SC2Spa

.. autosummary::
    :toctree: api

    pl.DrawGenes2
    pl.DrawCT1
    pl.DrawCT2
    pl.DrawCT3
    pl.DrawSVG
    pl.Superimpose
    pl.draw_cb
    
Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: SC2Spa.bm
.. currentmodule:: SC2Spa

.. autosummary::
    :toctree: api

    bm.BVMI
    bm.Vis_Euclidean

Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: SC2Spa.pp
.. currentmodule:: SC2Spa

.. autosummary::
    :toctree: api

    pp.MinMaxNorm
    pp.ReMMNorm
    pp.PolarTrans
    pp.RePolarTrans

Spatially Variable Gene Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: SC2Spa.sva
.. currentmodule:: SC2Spa

.. autosummary::
    :toctree: api

    sva.PrioritizeLPG
    sva.SelectGenes
    sva.SelectFeatures

Mutual Exclusivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: SC2Spa.me
.. currentmodule:: SC2Spa

.. autosummary::
    :toctree: api

    me.BME_stat
    me.calc_BME
    me.calc_BME_sub
    me.calc_DEEI
    me.DEEI_sub
    me.Count_Prob_EEI

