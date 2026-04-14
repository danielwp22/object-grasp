# Closed-loop grasp controller

## Problem

Build a closed-loop controller that can grasp a moving object: a ball rolling down a 15° ramp.

**Key challenge**: The object accelerates to >3.5 m/s while the robot has a velocity limit of 3.0 m/s, creating a critical time window where the grasp must succeed before the object becomes uncatchable.

**Constraints**:
- Noisy position measurements (σ = 1.2cm)
- 15ms sensor latency
- 2nd-order Cartesian plant dynamics (m·ẍ = u − b·ẋ)
- Physical grasp validation (no teleportation)

## Solution

A two-phase strategy combining **predictive planning** with **visual servoing**:

1. **APPROACH**: Intercept planner predicts where/when the object will be reachable. The end-effector moves toward the predicted intercept point.

2. **PRE_GRASP**: Visual servoing actively tracks the current object position (via Kalman filter) to compensate for prediction errors and achieve grasp conditions.

**Control architecture**: PID + Feedforward control with derivative filtering to handle noise while maintaining responsiveness.

## Results

**Grasp achieved at t=0.80s** with the following performance:

- **Spatial accuracy**: 5.6 cm from object center
- **Velocity matching**: <0.55 m/s relative velocity
- **Success rate**: Consistent grasp within physical validation criteria

### Trajectory

The end-effector (blue) approaches the predicted intercept zone, then switches to visual servoing to track the object (orange) down the ramp. The green star marks the successful grasp.

![Trajectory plot](assets/trajectory.png)

### Tracking error

The tracking error shows distinct phase behavior. During visual servoing (orange segment at t=0.6-0.8s), the controller maintains ~5.6cm steady-state error while matching the object's velocity.

![Tracking error](assets/tracking_error.png)

### End-effector speed

The robot accelerates to match the object's velocity, saturating at 3.0 m/s around t=1.3s. The grasp must occur at t=0.8s before this saturation makes the object uncatchable.

![EE speed](assets/ee_speed.png)

## Key findings

### Why visual servoing is essential

Initial "lock and wait" strategy (freeze intercept prediction and approach it) failed with:
- 9.2 cm spatial miss
- 2.5 m/s relative velocity at grasp attempt

Visual servoing (actively track current position) succeeded because it compensates for inevitable prediction errors from Kalman filter uncertainty and dynamic object motion.

### Why PID + Feedforward

**PID vs PD**: Tested comparison showed PID achieves 0.84s grasp vs 1.025s for PD (18% faster). The integral term eliminates steady-state error when tracking accelerating targets.

**Feedforward**: Model-based compensation (u_ff = m·a_des + b·v_des) reduces tracking error by providing bulk of control effort. Using 70% feedforward gain balances performance with model uncertainty.

**Derivative filter**: 10ms low-pass filter on D-term reduces chattering from velocity noise without sacrificing responsiveness.

### Tuning methodology

Systematically compared 5 tuning methods (Ziegler-Nichols, Cohen-Coon, IMC, Lambda, Optimization) using `tune_pid_advanced.py`.

**Critical lesson**: Mathematical optimization for ideal step responses can fail catastrophically on real tasks. "Optimal" gains (Kp=499, Ki=1, Kd=100) predicted perfect performance but completely failed due to noise amplification and saturation. Conservative empirical tuning (Kp=180, Ki=40, Kd=15) proved robust.

## Control parameters

- **PID gains**: Kp=180, Ki=40, Kd=15 (validated by Cohen-Coon method)
- **Derivative filter**: τ=10ms low-pass to reduce noise amplification
- **Feedforward gain**: 0.7 (70% model-based compensation)
- **Grasp tolerances**: 6cm spatial, 0.55 m/s velocity
- **Control frequency**: 200 Hz

## How to run

```bash
pip install -r requirements.txt
python grasp_controller.py
```

The interactive Plotly dashboard is saved to `outputs/grasp_controller.html` with full diagnostics including Kalman filter estimates, phase timeline, and performance metrics.

## Files

- **grasp_controller.py** - Main PID+Feedforward controller implementation
- **tune_pid_advanced.py** - Systematic comparison of 5 PID tuning methods
- **CONTROLLER_COMPARISON.md** - Detailed PID vs PD performance analysis

## System architecture

```
Object Simulator → Sensor (noise + latency) → Kalman Filter
                                                     ↓
                                              Intercept Planner
                                                     ↓
                                            Trajectory Generator
                                                     ↓
       Robot Plant ← PID Controller ← Desired trajectory
```

Pipeline executes at 200 Hz with predictive intercept planning in APPROACH, switching to visual servoing in PRE_GRASP to compensate for prediction errors.
