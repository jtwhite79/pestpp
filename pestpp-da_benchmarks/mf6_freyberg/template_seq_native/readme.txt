This directory contains a pestpp-da mf6 freyberg model variant.  The original model runs for one really long transient initial stress period, then two years of monthly stress periods - this model is documented in the pest++ v5 usgs report (White and others, 2021), with the one difference being that the initial stress period has been converted from steady state to transient to allow pestpp-da to drive the across the 25 stress periods without using any pre-(or post-)processors

To apply pestpp-da, the model was converted to a single stress period model and a parameter was added for the perlen variable in the mf6 tdis file. Additionally "dynamic states" were added to track the initial specified and final simulated groundwater levels for every active model cell as parameters and observations, respectively.  With these modifications, pestpp-da can replicate the transient time stepping of 25 stress periods of the original model.  

Historic observations are available for stress periods 2-13 at two groundwater level locations and one surface-water flow location. These were generated from the "truth" model described in the pest++ v5 usgs report. 

The last 12 stress periods are treated as a forecast period as no observations are assimilated but groundwater levels, surface water flows and surface-water/groundwater exchanges are monitored through observations in the control file.  In this way, users can experiement with different pestpp-da settings and compare the results of a sequential DA (iterative filter formualation) with pestpp-ies (e.g. iterative smoother formulation) results. 

To run this example, you need to have pestpp-da and mf6 (named "mf6") in your path or in this directory.
