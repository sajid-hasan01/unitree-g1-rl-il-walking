# AMASS B3 15-DOF Phase-BC IL Baseline

## Final Accepted Baseline

Dataset:
AMASS Retargeted G1 - ACCAD Female1Walking B3-walk1

Raw file:
experiments\amass_b3_15dof_il\raw\B3-walk1_poses_120_jpos.npz

Processed dataset:
experiments\amass_b3_15dof_il\processed\g1_amass_b3_walk1_il_15dof.npz

Model:
experiments\amass_b3_15dof_il_phase_bc\models\g1_amass_b3_phase_bc_15dof_best.pt

Algorithm:
Phase-conditioned Behavior Cloning

Control setup:
15 DOF lower body + waist

Input dimension:
9

Output dimension:
15

Training result:
Best epoch: 7976
Best validation MSE: 6.0109232435934246e-05

Visual evaluation result:
Mean MSE: 0.00003873
Mean MAE: 0.00310407
Max MSE: 0.00039898
Max MAE: 0.00880302
Max Abs Error: 0.07332537

Visual observation:
The predicted Phase-BC walking is smooth and close to the reference walking.

Conclusion:
This Phase-BC model is accepted as the improved AMASS B3 15-DOF IL baseline.
