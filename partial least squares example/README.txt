DF_hists_40bins_p1top99StrokeRange2_NOSPEED_v4_day1PLSR.mat - 2D histogram feature set data across 9 movement variables (end-point position, velocity and acceleration; elbow and shoulder joint pointion, velocity and acceleration; elbow and shoulder force, torque) for 22 stroke subjects.
Subject performed free exploration task while interacting with a planar robot. 

 
red_DesignnMat - histogram feature set consisting of a [22 subjects by 40*40 bins for 9 movement variables] matrix
		2D histogram data with 40 x 40 bins converted to 1 x 40*40 vector for each movement variable and concetenated.
		columns 1-40*40 - endpoint x,y position
		columns 40*40 + 1 - 40*40 + 40*40 - endpoint x,y velocity
		etc.


DF_y_Outcomes_day1 - y_[Outcome] - clinical and engineering outcome metrics for 22 stroke subjects
		     y_FM - Fugyl-meyyer scores
 