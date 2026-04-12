# Closed-loop grasp controller

## Problem
One paragraph — moving object, Kalman filter, PID, intercept planning.

## System architecture
Block diagram image here (inline in markdown)

## Results
Trajectory plot image + metrics table (overshoot, settling, SS error, grasp time)

## Design decisions  (this is what will impress)
- Why PID over PD: integral term needed to eliminate SS error on accelerating target
- Why PRE_GRASP locks p* and t*: prevents chasing a receding target
- Why metrics are computed on PRE_GRASP, not APPROACH: only phase with stationary setpoint
- Grippers/IK abstracted: scope decision, not oversight

## How to run
pip install -r requirements.txt
python grasp_controller.py

## What I'd do next (if not a time-boxed take-home)
- Real joint dynamics + IK
- Velocity matching in PRE_GRASP via dedicated gain schedule
- Gripper force-closure model
- 3D extension