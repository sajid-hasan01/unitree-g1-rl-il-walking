# Dynamic Validation Result - AMASS B3 15-DOF Phase-BC

## Baseline

Dataset:
AMASS B3-walk1

Policy:
experiments\amass_b3_15dof_il_phase_bc\models\g1_amass_b3_phase_bc_15dof_best.pt

Control:
15 DOF lower body + waist

Algorithm:
Phase-conditioned Behavior Cloning

## Visual Validation

The predicted Phase-BC motion was smooth and close to the reference replay.

Visual metrics:
Mean MSE: 0.00003873
Mean MAE: 0.00310407
Max MSE: 0.00039898
Max MAE: 0.00880302
Max Abs Error: 0.07332537

## Dynamic Validation

When the Phase-BC joint targets were executed in MuJoCo with the root free under physics, the robot started from a proper standing pose but eventually fell backward.

Test results:
- start_frame 90, action_scale 0.20: fell at step 187
- start_frame 90, reverse_time, action_scale 0.25: fell at step 163
- start_frame 130, action_scale 0.25: fell at step 157

## Interpretation

The Phase-BC policy is a strong kinematic imitation baseline, but it is open-loop and does not react to body pitch, base velocity, foot contact, slipping, or center-of-mass shift. Therefore, it cannot be considered a complete dynamic walking controller by itself.

## Final Decision

Keep AMASS B3 15-DOF Phase-BC as the accepted IL baseline.

Next step:
Add a balance feedback layer or residual RL controller on top of the Phase-BC baseline.
