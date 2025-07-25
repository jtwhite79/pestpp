This directory contains an mf6 freyberg variant almost identical to the model described in the pest++ v5 usgs report (the only difference being that the initial stress period was converted from steady state to a really long transient period).  

The control file "freyberg6_run_ies_pareto.pst" can be used to undertake predictive hypothesis testing around the simulated surface-water/groundwater exchange during the dry portion of the forecast period (the 17th stress period).  The only changes made to the control file to prepare for this analysis was the addition of two "++" args: da_weight_cycle_table and da_observation_cycle_table.  These two arg point to csv files that contain cycle-specific obsvals and weights.  The obsvals and weights for the historic observations are only replicated across the listed cycles using the values in the control file. The weight and/obsvals for the forecast of interest (observation "headwater_20172031") are varied across the cycles.  In this way, each cycle represents an iterative ensemble smoother analysis with varied values for forecast weight and obsval, essentially exploring the tradeoff between fitting historic observations and a desire for certain simulated forecast outcome.  

To run this example, you will need pestpp-da and mf6 (named "mf6") in your path or in this directory.

Enjoy!

