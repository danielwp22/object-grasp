"""
Closed-loop PID feedback controller for grasping a moving object.
Scenario: ball rolling down a 30° ramp.

Key design decisions:
  - APPROACH phase: continuously re-plans to updated KF intercept estimate
  - PRE_GRASP phase: locks onto a FIXED intercept point p* (frozen at transition)
      → prevents the EE from chasing a receding target
  - GRASP triggered by: position close enough AND time-to-intercept < window
  - POST_GRASP: minimum-jerk lift to a fixed world pose above grasp point
  - Plant: 2nd-order Cartesian (closed-loop robot abstraction, no joint dynamics)
  - Grippers: binary event (force-closure not modelled)

Pipeline each timestep (200 Hz outer loop):
  ObjectSimulator → KalmanFilter → InterceptPlanner
  → TrajectoryGenerator → PIDController → RobotPlant → DataLogger
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Simulation constants
# ─────────────────────────────────────────────────────────────────
DT              = 0.005         # control loop timestep (s) — 200 Hz
T_MAX           = 4.0           # max simulation time (s)
G               = 9.81          # m/s²
THETA           = np.radians(15)
RAMP_DIR        = np.array([np.cos(-THETA), np.sin(-THETA)])  # unit vec downhill
OBJ_A_RAMP      = G * np.sin(THETA)   # 4.905 m/s²

# Sensor model
MEAS_STD        = 0.012         # position noise std (m)
LATENCY_STEPS   = 3             # fixed sensor delay in timesteps

# Robot plant (Cartesian 2nd-order: m·ẍ = u − b·ẋ)
EE_MASS         = 1.0           # kg
EE_DAMPING      = 5.0           # N·s/m
EE_V_MAX        = 3.0           # m/s
EE_A_MAX        = EE_V_MAX / 0.10  # m/s²

# Intercept planner
PLAN_HORIZON    = 1.2           # max forward look-ahead (s)

# Grasp geometry — offsets relative to object CoM, world frame
PREGRASP_OFFSET = np.array([0.0,  0.10])
GRASP_OFFSET    = np.array([0.0,  0.00])
POSTGRASP_LIFT  = np.array([0.0,  0.30])

# Phase transition thresholds
PREGRASP_DIST   = 0.10          # m — enter PRE_GRASP
GRASP_DIST      = 0.015         # m — position close enough
GRASP_REL_V_MAX = 0.20          # m/s — relative speed for closing grasp
GRASP_T_WINDOW  = 0.12          # s — time-to-intercept for grasp trigger

# Grasp success criteria (physical validation)
GRASP_SPATIAL_TOL = 0.06        # m — EE must be within this distance of object (6cm)
GRASP_VELOCITY_TOL = 0.55       # m/s — relative velocity must be below this (relaxed for fast objects)

# PID + Feedforward gains (empirical tuning, close to Cohen-Coon recommendation)
KP = np.array([180.0, 180.0])
KI = np.array([40.0,  40.0])
KD = np.array([15.0,  15.0])  # Conservative Kd to minimize noise amplification
I_CLAMP = 0.25
D_FILTER_TAU = 0.01  # Derivative filter time constant (s) - low-pass to reduce noise


# ─────────────────────────────────────────────────────────────────
class Phase(Enum):
    APPROACH   = auto()
    PRE_GRASP  = auto()
    GRASP      = auto()
    POST_GRASP = auto()
    DONE       = auto()

PHASE_COLORS = {
    "APPROACH":   "#378ADD",
    "PRE_GRASP":  "#BA7517",
    "GRASP":      "#1D9E75",
    "POST_GRASP": "#D85A30",
    "DONE":       "#888780",
}

@dataclass
class GraspSpec:
    pregrasp:  np.ndarray = field(default_factory=lambda: PREGRASP_OFFSET.copy())
    grasp:     np.ndarray = field(default_factory=lambda: GRASP_OFFSET.copy())
    postgrasp: np.ndarray = field(default_factory=lambda: POSTGRASP_LIFT.copy())


# ─────────────────────────────────────────────────────────────────
# 1. Object Simulator
# ─────────────────────────────────────────────────────────────────
class ObjectSimulator:
    """Ground-truth constant-acceleration kinematics on a frictionless ramp."""
    def __init__(self, start_pos: np.ndarray, v0: float = 0.0):
        self.origin = start_pos.astype(float).copy()
        self.v0     = v0
        self.rng    = np.random.default_rng(42)
        self.grasped = False
        self.grasp_pos = None
        self.grasp_vel = None

    def true_state(self, t: float):
        if self.grasped and self.grasp_pos is not None:
            return self.grasp_pos.copy(), self.grasp_vel.copy()
        s   = self.v0 * t + 0.5 * OBJ_A_RAMP * t**2
        v_s = self.v0 + OBJ_A_RAMP * t
        return self.origin + s * RAMP_DIR, v_s * RAMP_DIR

    def measure(self, t: float) -> np.ndarray:
        """Additive Gaussian noise on position."""
        pos, _ = self.true_state(t)
        return pos + self.rng.normal(0, MEAS_STD, 2)

    def on_ramp(self, t: float) -> bool:
        if self.grasped:
            return True
        pos, _ = self.true_state(t)
        return bool(pos[1] > 0.0)

    def attach(self, pos: np.ndarray, vel: np.ndarray):
        self.grasped = True
        self.grasp_pos = pos.astype(float).copy()
        self.grasp_vel = vel.astype(float).copy()

    def update_attached(self, pos: np.ndarray, vel: np.ndarray):
        if not self.grasped:
            return
        self.grasp_pos = pos.astype(float).copy()
        self.grasp_vel = vel.astype(float).copy()


# ─────────────────────────────────────────────────────────────────
# 2. Kalman Filter  (state = [x, y, vx, vy, ax, ay])
# ─────────────────────────────────────────────────────────────────
class KalmanFilter:
    """
    Linear KF with constant-acceleration motion model.
    After update, predicts forward LATENCY_STEPS·DT to compensate for
    known sensor delay, returning an estimate of the current state.
    """
    def __init__(self, init_pos: np.ndarray):
        self.x = np.zeros(6)
        self.x[:2] = init_pos.copy()
        dt = DT
        self.F = np.eye(6)
        self.F[0,2]=dt; self.F[0,4]=0.5*dt**2
        self.F[1,3]=dt; self.F[1,5]=0.5*dt**2
        self.F[2,4]=dt; self.F[3,5]=dt
        self.H = np.zeros((2, 6))
        self.H[0,0] = 1.0; self.H[1,1] = 1.0
        q = 0.08
        self.Q = np.diag([1e-5, 1e-5, q*0.5, q*0.5, q, q])
        self.R = np.eye(2) * MEAS_STD**2
        self.P = np.eye(6) * 0.5

    def _step(self, x, P, n=1):
        for _ in range(n):
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
        return x, P

    def update(self, z: np.ndarray) -> np.ndarray:
        self.x, self.P = self._step(self.x, self.P)
        S  = self.H @ self.P @ self.H.T + self.R
        K  = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ (z - self.H @ self.x)
        self.P  = (np.eye(6) - K @ self.H) @ self.P
        x_now, _ = self._step(self.x.copy(), self.P.copy(), LATENCY_STEPS)
        return x_now

    def predict_at(self, dt_ahead: float):
        steps = max(1, int(round(dt_ahead / DT)))
        x_pred, _ = self._step(self.x.copy(), self.P.copy(), steps)
        return x_pred[:2], x_pred[2:4]


# ─────────────────────────────────────────────────────────────────
# 3. Intercept Planner
# ─────────────────────────────────────────────────────────────────
class InterceptPlanner:
    """
    Finds t* via bisection on:
        residual(t*) = robot_travel_time(|p*(t*) - ee|) - t*

    travel_time uses a trapezoidal velocity profile (accel to v_max, cruise,
    decel) — minimum-time primitive for a speed-limited system.
    """
    def travel_time(self, dist: float) -> float:
        if dist < 1e-6: return 0.0
        d_ramp = EE_V_MAX**2 / (2 * EE_A_MAX)
        if dist <= 2 * d_ramp:
            return 2 * np.sqrt(dist / EE_A_MAX)
        return 2 * (EE_V_MAX / EE_A_MAX) + (dist - 2 * d_ramp) / EE_V_MAX

    def solve(self, kf: KalmanFilter, ee_pos: np.ndarray):
        def residual(ts):
            p, _ = kf.predict_at(ts)
            return self.travel_time(np.linalg.norm(p - ee_pos)) - ts
        t_lo, t_hi = DT, PLAN_HORIZON
        r_lo = residual(t_lo)
        if r_lo * residual(t_hi) > 0:
            t_star = t_hi
        else:
            for _ in range(25):
                t_mid = 0.5 * (t_lo + t_hi)
                if residual(t_mid) * r_lo < 0: t_hi = t_mid
                else: t_lo = t_mid; r_lo = residual(t_mid)
            t_star = 0.5 * (t_lo + t_hi)
        p_star, v_star = kf.predict_at(t_star)
        return t_star, p_star, v_star


# ─────────────────────────────────────────────────────────────────
# 4. Trajectory Generator  (minimum-jerk polynomial)
# ─────────────────────────────────────────────────────────────────
class TrajectoryGenerator:
    """
    x(tau) = x0 + dx*(10t^3 - 15t^4 + 6t^5), tau in [0,1]
    Zero velocity and acceleration at endpoints — smooth, no impulsive forces.

    APPROACH:   targets pre-grasp pose above predicted intercept p*
    PRE_GRASP:  actively tracks current object position and velocity (visual servoing)
                to close the gap created by prediction errors
    GRASP:      rigidly tracks object CoM (gripper closed)
    POST_GRASP: min-jerk lift to fixed pose above grasp point
    """
    def min_jerk(self, x0, xf, t, T):
        """Minimum jerk trajectory with position, velocity, and acceleration."""
        tau = np.clip(t / T, 0.0, 1.0)
        dx = xf - x0
        T_safe = max(T, 1e-6)

        pos = x0 + dx * (10*tau**3 - 15*tau**4 + 6*tau**5)
        vel = dx / T_safe * (30*tau**2 - 60*tau**3 + 30*tau**4)
        acc = dx / T_safe**2 * (60*tau - 180*tau**2 + 120*tau**3)

        return pos, vel, acc

    def desired(self, phase, t_in, dur, ee0, kf_pos, kf_vel, kf_acc, locked_pstar, spec):
        """Generate desired trajectory with position, velocity, and acceleration."""
        # Physics-based acceleration for the ball (constant, known from ramp dynamics)
        ball_acc = OBJ_A_RAMP * RAMP_DIR

        if phase == Phase.APPROACH:
            # Target predicted intercept point, not current object position.
            # This drives EE toward where the object WILL BE, not where it IS.
            return self.min_jerk(ee0, locked_pstar + spec.pregrasp, t_in, dur)

        elif phase == Phase.PRE_GRASP:
            # Visual servoing: actively track current object position and velocity.
            # Smooth transition using S-curve (3rd order polynomial) to minimize jerk
            # and reduce transient tracking error spike at phase entry.
            # Use physics-based acceleration (smooth, known) instead of noisy KF estimate.
            blend_time = 0.05  # 10 timesteps @ 200Hz - smooth S-curve transition
            if t_in < blend_time:
                # S-curve: tau^3 * (6*tau^2 - 15*tau + 10)
                tau = t_in / blend_time
                alpha = tau**3 * (6*tau**2 - 15*tau + 10)  # Smooth 0→1 transition
                target_pos = (1 - alpha) * (locked_pstar + spec.pregrasp) + alpha * (kf_pos + spec.grasp)
                target_vel = alpha * kf_vel
                target_acc = alpha * ball_acc  # Use physics model, not noisy KF
            else:
                target_pos = kf_pos + spec.grasp
                target_vel = kf_vel
                target_acc = ball_acc  # Use physics model, not noisy KF
            return target_pos, target_vel, target_acc

        elif phase == Phase.GRASP:
            # Object is grasped - hold it stationary (zero velocity/acceleration)
            # Prevents drift during the grasp-hold period before lifting
            return kf_pos + spec.grasp, np.zeros(2), np.zeros(2)

        elif phase == Phase.POST_GRASP:
            # Lift from actual grasp position (ee0), not from old prediction (locked_pstar)
            lift_target = ee0 + spec.postgrasp
            return self.min_jerk(ee0, lift_target, t_in, max(dur, 1.0))

        return ee0.copy(), np.zeros(2), np.zeros(2)


# ─────────────────────────────────────────────────────────────────
# 5. PID Controller  (2D, anti-windup)
# ─────────────────────────────────────────────────────────────────
class PIDController:
    """
    Feedforward + Feedback control:
    u = u_feedforward + u_feedback

    Feedforward (model-based):
      u_ff = m*a_desired + b*v_desired
      Inverts plant dynamics to anticipate required control

    Feedback (error-driven):
      u_fb = Kp*e_pos + Ki*integral(e_pos) + Kd*filtered(de_pos/dt + e_vel)
      Corrects for model uncertainties and disturbances

    Derivative filtering:
      Low-pass filter on derivative term to reduce high-frequency noise amplification
      d_filtered[k] = α*d_raw[k] + (1-α)*d_filtered[k-1]
      where α = DT/(DT + τ), τ = filter time constant

    Benefits:
    - Feedforward provides bulk of control → smaller tracking errors
    - Feedback handles residual errors → robustness
    - Derivative filter reduces chattering from sensor noise
    - Combined: fast response + accuracy + robustness
    """
    def __init__(self, use_feedforward=True, ff_gain=1.0):
        self.integral   = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.d_filtered = np.zeros(2)  # Filtered derivative term
        self.use_feedforward = use_feedforward
        self.ff_gain = ff_gain  # Feedforward gain (0-1), allows detuning if model uncertain

    def reset(self):
        self.integral   = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.d_filtered = np.zeros(2)

    def compute(self, des_pos, des_vel, ee_pos, ee_vel, des_acc=None):
        """
        Compute control with feedforward + feedback.

        Args:
            des_pos: desired position
            des_vel: desired velocity
            ee_pos: actual position
            ee_vel: actual velocity
            des_acc: desired acceleration (optional, for feedforward)
        """
        # Feedback control (error-driven)
        e_pos = des_pos - ee_pos
        e_vel = des_vel - ee_vel
        self.integral  = np.clip(self.integral + e_pos * DT, -I_CLAMP, I_CLAMP)

        # Raw derivative term
        d_raw = (e_pos - self.prev_error) / DT + e_vel
        self.prev_error = e_pos.copy()

        # Low-pass filter on derivative to reduce noise amplification
        # First-order discrete filter: α = DT/(DT + τ)
        alpha = DT / (DT + D_FILTER_TAU)
        self.d_filtered = alpha * d_raw + (1 - alpha) * self.d_filtered

        u_feedback = KP * e_pos + KI * self.integral + KD * self.d_filtered

        # Feedforward control (model-based)
        if self.use_feedforward and des_acc is not None:
            # Invert plant dynamics: u = m*a + b*v
            u_feedforward = EE_MASS * des_acc + EE_DAMPING * des_vel
            u_feedforward *= self.ff_gain  # Allow detuning
        else:
            u_feedforward = np.zeros(2)

        return u_feedforward + u_feedback


# ─────────────────────────────────────────────────────────────────
# 6. Robot Plant  (2nd-order Cartesian)
# ─────────────────────────────────────────────────────────────────
class RobotPlant:
    """
    m*x_ddot = u - b*x_dot   (semi-implicit Euler)

    Abstracts the closed-loop robot (inner joint controller already running).
    Damping b models the inner loop's velocity resistance, giving realistic
    overshoot and settling without joint-level simulation.

    Grippers are a binary event triggered by the phase state machine —
    no contact dynamics or force closure modelled here.
    """
    def __init__(self, init_pos: np.ndarray):
        self.pos = init_pos.astype(float).copy()
        self.vel = np.zeros(2)

    def step(self, u: np.ndarray):
        acc      = (u - EE_DAMPING * self.vel) / EE_MASS
        self.vel += acc * DT
        speed     = np.linalg.norm(self.vel)
        if speed > EE_V_MAX:
            self.vel *= EE_V_MAX / speed
        self.pos += self.vel * DT
        return self.pos.copy(), self.vel.copy()


# ─────────────────────────────────────────────────────────────────
# 7. Data Logger
# ─────────────────────────────────────────────────────────────────
class DataLogger:
    def __init__(self):
        self.data = {k: [] for k in [
            "t","obj_pos","noisy_pos","kf_pos","ee_pos","ee_vel",
            "des_pos","pos_error","phase","t_star","p_star"]}

    def log(self, **kw):
        for k, v in kw.items():
            self.data[k].append(v.copy() if isinstance(v, np.ndarray) else v)

    def arrays(self):
        out = {}
        for k, v in self.data.items():
            try:    out[k] = np.array(v)
            except: out[k] = np.array(v, dtype=object)
        return out


# ─────────────────────────────────────────────────────────────────
# 8. Main simulation loop
# ─────────────────────────────────────────────────────────────────
def run_simulation():
    obj_start = np.array([0.0,  1.5])
    ee_start  = np.array([2.07, 1.45])  # above predicted intercept zone (~t=1.3s)

    obj   = ObjectSimulator(obj_start)
    kf    = KalmanFilter(obj_start)
    plan  = InterceptPlanner()
    traj  = TrajectoryGenerator()
    pid   = PIDController(use_feedforward=True, ff_gain=0.7)  # Reduced gain to minimize chattering
    robot = RobotPlant(ee_start)
    spec  = GraspSpec()
    log   = DataLogger()

    phase        = Phase.APPROACH
    phase_t0     = 0.0
    phase_ee0    = ee_start.copy()
    locked_pstar = obj_start.copy()   # frozen p* at PRE_GRASP entry
    locked_tstar  = PLAN_HORIZON         # frozen t* at PRE_GRASP entry
    pregrasp_entry_t = 0.0               # time when we entered PRE_GRASP
    grasp_t      = None

    # Warm up KF — enough steps for velocity/accel estimates to converge
    for i in range(40):
        kf.update(obj.measure(i * DT))

    for step in range(int(T_MAX / DT)):
        t = step * DT
        if not obj.on_ramp(t):
            break

        # 1. Sense (delayed)
        z = obj.measure(max(0.0, t - LATENCY_STEPS * DT))
        obj_pos, obj_vel = obj.true_state(t)

        # 2. Estimate
        x_hat  = kf.update(z)
        kf_pos = x_hat[:2]
        kf_vel = x_hat[2:4]
        kf_acc = x_hat[4:6]  # Extract acceleration estimate from KF state

        # 3. Plan
        ee_pos, ee_vel = robot.pos.copy(), robot.vel.copy()
        t_star, p_star, v_star = plan.solve(kf, ee_pos)

        # 4. Trajectory (now returns pos, vel, acc for feedforward control)
        t_in = t - phase_t0
        dur  = max(locked_tstar if phase == Phase.PRE_GRASP else t_star, 0.15)
        des_pos, des_vel, des_acc = traj.desired(
            phase, t_in, dur, phase_ee0,
            kf_pos, kf_vel, kf_acc, locked_pstar, spec)

        # 5. Feedforward + Feedback control
        u = pid.compute(des_pos, des_vel, ee_pos, ee_vel, des_acc)

        # 6. Plant
        ee_pos, ee_vel = robot.step(u)
        if obj.grasped:
            obj.update_attached(ee_pos, ee_vel)
            obj_pos, obj_vel = obj.true_state(t)

        # 7. Phase machine
        t_in_phase = t - phase_t0
        # Distance to predicted intercept (for entering PRE_GRASP)
        dist_pregrasp = np.linalg.norm(ee_pos - (p_star + spec.pregrasp))
        # Distance to current object position (for tracking)
        dist_to_obj = np.linalg.norm(ee_pos - obj_pos)
        rel_speed = np.linalg.norm(ee_vel - obj_vel)

        def enter(p):
            nonlocal phase, phase_t0, phase_ee0, locked_pstar, pregrasp_entry_t
            phase = p; phase_t0 = t; phase_ee0 = ee_pos.copy()
            # Don't reset PID integral on PRE_GRASP transition to avoid losing
            # accumulated compensation. Only reset on major transitions.
            if p in [Phase.APPROACH, Phase.POST_GRASP]:
                pid.reset()

        if phase == Phase.APPROACH:
            locked_pstar = p_star.copy()   # keep updating best intercept estimate
            if dist_pregrasp < PREGRASP_DIST:
                locked_tstar = t_star      # also freeze the time budget
                pregrasp_entry_t = t       # record when we locked the intercept
                enter(Phase.PRE_GRASP)

        elif phase == Phase.PRE_GRASP:
            # Visual servoing approach: actively track the object until close enough.
            # Trigger grasp when physical conditions are met:
            # 1. EE is spatially close to object
            # 2. Velocity is matched
            actual_dist = np.linalg.norm(ee_pos - obj_pos)
            actual_rel_vel = np.linalg.norm(ee_vel - obj_vel)

            if actual_dist < GRASP_SPATIAL_TOL and actual_rel_vel < GRASP_VELOCITY_TOL:
                # Physical conditions met - close gripper!
                enter(Phase.GRASP)
                grasp_t = t
                obj.attach(obj_pos, obj_vel)
            elif t_in_phase > 2.0:
                # Timeout - couldn't achieve grasp conditions
                enter(Phase.DONE)

        elif phase == Phase.GRASP and t_in_phase > 0.25:
            enter(Phase.POST_GRASP)

        elif phase == Phase.POST_GRASP and t_in_phase > 1.8:
            enter(Phase.DONE)

        # 8. Log
        log.log(t=t, obj_pos=obj_pos, noisy_pos=z, kf_pos=kf_pos,
                ee_pos=ee_pos, ee_vel=ee_vel, des_pos=des_pos,
                pos_error=float(np.linalg.norm(des_pos - ee_pos)),
                phase=phase.name, t_star=float(t_star), p_star=p_star)

        if phase == Phase.DONE:
            break

    return log, grasp_t


# ─────────────────────────────────────────────────────────────────
# 9. Performance Metrics
# ─────────────────────────────────────────────────────────────────
def compute_metrics(data: dict, grasp_t) -> dict:
    """
    Metrics computed on the PRE_GRASP phase — the step response to a fixed
    target (locked_pstar). This is the correct phase for classical PID metrics
    because the setpoint is stationary during PRE_GRASP.

    APPROACH is excluded: the target (p_star) moves every step, making
    classical overshoot/settling undefined.

    Overshoot:          (peak - SS) / SS * 100%
    Settling time:      first t where error stays within 5% band of SS
    Steady-state error: mean error in final 20% of PRE_GRASP
    """
    t, err, phases = data["t"], data["pos_error"], data["phase"]

    # Use PRE_GRASP phase (fixed setpoint = locked_pstar)
    mask = phases == "PRE_GRASP"
    if mask.sum() < 5:
        return {"grasp_time_s": round(float(grasp_t), 3) if grasp_t else None}

    t_pg, e_pg = t[mask], err[mask]
    n_ss      = max(1, len(e_pg) // 5)
    e_ss      = float(np.mean(e_pg[-n_ss:]))
    e_max     = float(e_pg.max())
    overshoot = max(0.0, (e_max - e_ss) / (e_ss + 1e-9) * 100)
    band      = max(0.05 * e_ss, 1e-3)
    t_settle  = float(t_pg[-1])
    for i in range(len(e_pg)):
        if np.all(np.abs(e_pg[i:] - e_ss) <= band):
            t_settle = float(t_pg[i])
            break
    return {
        "overshoot_pct":        round(overshoot, 1),
        "settling_time_s":      round(t_settle - float(t_pg[0]), 3),
        "steady_state_error_m": round(e_ss, 4),
        "grasp_time_s":         round(float(grasp_t), 3) if grasp_t else None,
    }


# ─────────────────────────────────────────────────────────────────
# 10. Plotly Visualisation
# ─────────────────────────────────────────────────────────────────
def plot_results(logger: DataLogger, grasp_t):
    data    = logger.arrays()
    metrics = compute_metrics(data, grasp_t)
    t, phases = data["t"], data["phase"]
    obj, ee   = data["obj_pos"], data["ee_pos"]
    kfp, noisy = data["kf_pos"], data["noisy_pos"]
    err, ee_v  = data["pos_error"], data["ee_vel"]
    speed = np.linalg.norm(ee_v, axis=1)

    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"type":"xy"},    {"type":"xy"}],
               [{"type":"xy"},    {"type":"xy"}],
               [{"type":"xy"},    {"type":"table"}]],
        subplot_titles=[
            "Trajectory (world frame)",
            "Tracking error over time",
            "KF estimate vs ground truth (x-axis)",
            "End-effector speed",
            "Phase timeline",
            "Performance metrics",
        ],
        row_heights=[0.42, 0.33, 0.25],
        vertical_spacing=0.10,
        horizontal_spacing=0.12,
    )

    # [1,1] 2D trajectory
    ramp_s = np.linspace(0, 3.2, 80)
    fig.add_trace(go.Scatter(
        x=ramp_s*RAMP_DIR[0], y=1.5+ramp_s*RAMP_DIR[1],
        name="Ramp", mode="lines",
        line=dict(color="#444441", width=1, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=noisy[:,0], y=noisy[:,1], name="Measurement",
        mode="markers", marker=dict(color="#D85A30", size=2, opacity=0.3)),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=obj[:,0], y=obj[:,1], name="Object (truth)",
        line=dict(color="#D85A30", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=kfp[:,0], y=kfp[:,1], name="KF estimate",
        line=dict(color="#BA7517", width=1.5, dash="dot")), row=1, col=1)
    for ph, col in PHASE_COLORS.items():
        m = phases == ph
        if m.any():
            fig.add_trace(go.Scatter(
                x=ee[m,0], y=ee[m,1], name=f"EE — {ph}",
                mode="lines", line=dict(color=col, width=2.5)), row=1, col=1)
    if grasp_t is not None:
        idx = np.argmin(np.abs(t - grasp_t))
        fig.add_trace(go.Scatter(
            x=[ee[idx,0]], y=[ee[idx,1]], name="Grasp event ★",
            mode="markers", marker=dict(
                symbol="star", size=22, color="#1D9E75",
                line=dict(color="white", width=1.5))), row=1, col=1)

    # [1,2] Tracking error by phase
    for ph, col in PHASE_COLORS.items():
        m = phases == ph
        if m.any():
            fig.add_trace(go.Scatter(
                x=t[m], y=err[m]*100, name=f"err — {ph}",
                mode="lines", line=dict(color=col, width=1.5),
                showlegend=False), row=1, col=2)

    # Annotations anchored to PRE_GRASP (step response to fixed target)
    pg_mask = phases == "PRE_GRASP"
    e_ss_cm = metrics.get("steady_state_error_m", 0) * 100
    if e_ss_cm > 0 and pg_mask.any():
        t_pg_start = float(t[pg_mask][0])
        t_pg_end   = float(t[pg_mask][-1])
        # SS band shaded across full plot width
        fig.add_hrect(y0=0, y1=e_ss_cm * 1.05, fillcolor="#1D9E75",
                      opacity=0.07, line_width=0, row=1, col=2)
        # SS error label inside PRE_GRASP window
        fig.add_annotation(
            x=t_pg_start + (t_pg_end - t_pg_start) * 0.5, y=e_ss_cm * 2.8,
            text=f"SS ≈ {e_ss_cm:.1f} cm",
            showarrow=False, font=dict(size=11, color="#1D9E75"), row=1, col=2)
        # Overshoot annotation at peak
        e_pg = err[pg_mask] * 100
        peak_idx_local = int(np.argmax(e_pg))
        t_peak = float(t[pg_mask][peak_idx_local])
        e_peak = float(e_pg[peak_idx_local])
        ov = metrics.get("overshoot_pct", 0)
        fig.add_annotation(
            x=t_peak, y=e_peak + 1.5,
            text=f"peak  {ov:.0f}% OS",
            showarrow=True, arrowhead=2, arrowcolor="#BA7517",
            font=dict(size=10, color="#BA7517"), row=1, col=2)
        # Settling time vline (anchored to PRE_GRASP start)
        ts = metrics.get("settling_time_s")
        if ts is not None:
            fig.add_vline(x=t_pg_start + ts, line_dash="dot",
                          line_color="#378ADD",
                          annotation_text=f"settle {ts:.2f}s",
                          row=1, col=2)
        # Grasp event vline
        if grasp_t is not None:
            fig.add_vline(x=grasp_t, line_dash="dash",
                          line_color="#1D9E75",
                          annotation_text="grasp",
                          row=1, col=2)

    # [2,1] KF vs truth (x component)
    fig.add_trace(go.Scatter(x=t, y=noisy[:,0]*100, name="Measured x",
        mode="markers", marker=dict(color="#D85A30", size=2, opacity=0.25)),
        row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=obj[:,0]*100, name="True x",
        line=dict(color="#D85A30", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=kfp[:,0]*100, name="KF estimate x",
        line=dict(color="#BA7517", width=1.5, dash="dot")), row=2, col=1)

    # [2,2] EE speed
    fig.add_trace(go.Scatter(x=t, y=speed, name="|v_ee|",
        line=dict(color="#378ADD", width=1.5)), row=2, col=2)
    fig.add_hline(y=EE_V_MAX, line_dash="dash", line_color="#666",
        annotation_text=f"v_max = {EE_V_MAX} m/s", row=2, col=2)

    # [3,1] Phase Gantt
    segs = []
    for i, ph in enumerate(phases):
        if i == 0 or ph != phases[i-1]:
            segs.append([ph, float(t[i])])
    segs.append(["END", float(t[-1])])
    for k in range(len(segs)-1):
        ph, t0 = segs[k]; t1 = segs[k+1][1]
        fig.add_trace(go.Bar(
            x=[t1-t0], y=[ph], base=[t0], orientation="h",
            marker_color=PHASE_COLORS.get(ph, "#888780"),
            opacity=0.85, showlegend=False), row=3, col=1)

    # [3,2] Metrics table
    gt = metrics.get("grasp_time_s")
    rows_k = ["Overshoot","Settling time","Steady-state error",
              "Grasp event","KP / KI / KD","Plant mass / damping"]
    ov = metrics.get("overshoot_pct")
    rows_v = [
        f"{ov:.1f} %" if ov is not None else "—",
        f"{metrics.get('settling_time_s', 0):.3f} s",
        f"{metrics.get('steady_state_error_m', 0)*100:.2f} cm",
        f"t = {gt:.3f} s" if gt else "not achieved",
        f"{KP[0]} / {KI[0]} / {KD[0]}",
        f"{EE_MASS} kg  /  {EE_DAMPING} N·s/m",
    ]
    fig.add_trace(go.Table(
        header=dict(values=["Metric","Value"],
            fill_color="#3C3489", font=dict(color="white",size=12), align="left"),
        cells=dict(values=[rows_k, rows_v],
            fill_color="#1a1a1a", font=dict(color="#c2c0b6",size=12), align="left")),
        row=3, col=2)

    status = f"✓  grasp at t = {grasp_t:.3f} s" if grasp_t else "✗  no grasp achieved"
    fig.update_layout(
        title=f"Closed-loop PID grasp controller — ball on 30° ramp  ({status})",
        height=980, width=1400, template="plotly_dark",
        legend=dict(orientation="h", y=-0.08, font=dict(size=10)),
        margin=dict(l=70, r=40, t=80, b=80),
    )
    for r, c, xl, yl in [(1,1,"x (m)","y (m)"),(1,2,"time (s)","error (cm)"),
                          (2,1,"time (s)","x position (cm)"),(2,2,"time (s)","speed (m/s)"),
                          (3,1,"time (s)","")]:
        fig.update_xaxes(title_text=xl, row=r, col=c)
        fig.update_yaxes(title_text=yl, row=r, col=c)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "grasp_controller.html"
    fig.write_html(str(out))

    assets_dir = Path("assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    png_out = assets_dir / "grasp_controller.png"
    try:
        fig.write_image(str(png_out), width=1400, height=980, scale=2)
        _save_separate_figs(assets_dir, t, phases, obj, ee, kfp, noisy, err, speed, metrics, grasp_t)
        print(f"Saved → {png_out}")
    except Exception as exc:
        print(f"Skipping PNG export (install kaleido to enable): {exc}")

    print(f"Saved → {out}")
    print(f"Metrics: {metrics}")
    print(f"Grasp:   {'t = '+str(grasp_t)+'s' if grasp_t else 'NOT ACHIEVED'}")


def _save_separate_figs(assets_dir, t, phases, obj, ee, kfp, noisy, err, speed, metrics, grasp_t):
    def _save(fig, name, width=1200, height=700):
        fig.update_layout(template="plotly_white", margin=dict(l=60, r=30, t=60, b=60))
        fig.write_image(str(assets_dir / name), width=width, height=height, scale=2)

    # Trajectory
    fig_traj = go.Figure()
    ramp_s = np.linspace(0, 3.2, 80)
    fig_traj.add_trace(go.Scatter(
        x=ramp_s*RAMP_DIR[0], y=1.5+ramp_s*RAMP_DIR[1],
        name="Ramp", mode="lines",
        line=dict(color="#666", width=1, dash="dash")))
    fig_traj.add_trace(go.Scatter(
        x=noisy[:,0], y=noisy[:,1], name="Measurement",
        mode="markers", marker=dict(color="#D85A30", size=3, opacity=0.25)))
    fig_traj.add_trace(go.Scatter(
        x=obj[:,0], y=obj[:,1], name="Object (truth)",
        line=dict(color="#D85A30", width=2.5)))
    fig_traj.add_trace(go.Scatter(
        x=kfp[:,0], y=kfp[:,1], name="KF estimate",
        line=dict(color="#BA7517", width=1.5, dash="dot")))
    fig_traj.add_trace(go.Scatter(
        x=ee[:,0], y=ee[:,1], name="End-effector",
        line=dict(color="#378ADD", width=2.5)))
    if grasp_t is not None:
        idx = np.argmin(np.abs(t - grasp_t))
        fig_traj.add_trace(go.Scatter(
            x=[ee[idx,0]], y=[ee[idx,1]], name="Grasp",
            mode="markers", marker=dict(symbol="star", size=16, color="#1D9E75",
                                         line=dict(color="white", width=1))))
    fig_traj.update_layout(title="Trajectory (world frame)")
    fig_traj.update_xaxes(title_text="x (m)")
    fig_traj.update_yaxes(title_text="y (m)")
    _save(fig_traj, "trajectory.png", width=1100, height=800)

    # Tracking error
    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(
        x=t, y=err*100, name="Tracking error",
        line=dict(color="#378ADD", width=2)))
    if grasp_t is not None:
        fig_err.add_vline(x=grasp_t, line_dash="dash",
                          line_color="#1D9E75", annotation_text="grasp")
    fig_err.update_layout(title="Tracking error vs time")
    fig_err.update_xaxes(title_text="time (s)")
    fig_err.update_yaxes(title_text="error (cm)")
    _save(fig_err, "tracking_error.png")

    # KF vs truth (x)
    fig_kf = go.Figure()
    fig_kf.add_trace(go.Scatter(x=t, y=noisy[:,0]*100, name="Measured x",
        mode="markers", marker=dict(color="#D85A30", size=3, opacity=0.25)))
    fig_kf.add_trace(go.Scatter(x=t, y=obj[:,0]*100, name="True x",
        line=dict(color="#D85A30", width=2)))
    fig_kf.add_trace(go.Scatter(x=t, y=kfp[:,0]*100, name="KF estimate x",
        line=dict(color="#BA7517", width=1.5, dash="dot")))
    fig_kf.update_layout(title="KF estimate vs ground truth (x-axis)")
    fig_kf.update_xaxes(title_text="time (s)")
    fig_kf.update_yaxes(title_text="x position (cm)")
    _save(fig_kf, "kf_vs_truth.png")

    # EE speed
    fig_speed = go.Figure()
    fig_speed.add_trace(go.Scatter(x=t, y=speed, name="|v_ee|",
        line=dict(color="#378ADD", width=2)))
    fig_speed.add_hline(y=EE_V_MAX, line_dash="dash", line_color="#666",
        annotation_text=f"v_max = {EE_V_MAX} m/s")
    fig_speed.update_layout(title="End-effector speed")
    fig_speed.update_xaxes(title_text="time (s)")
    fig_speed.update_yaxes(title_text="speed (m/s)")
    _save(fig_speed, "ee_speed.png")

    # Phase timeline
    fig_phase = go.Figure()
    segs = []
    for i, ph in enumerate(phases):
        if i == 0 or ph != phases[i-1]:
            segs.append([ph, float(t[i])])
    segs.append(["END", float(t[-1])])
    for k in range(len(segs)-1):
        ph, t0 = segs[k]; t1 = segs[k+1][1]
        fig_phase.add_trace(go.Bar(
            x=[t1-t0], y=[ph], base=[t0], orientation="h",
            marker_color=PHASE_COLORS.get(ph, "#888780"),
            opacity=0.85, showlegend=False))
    fig_phase.update_layout(title="Phase timeline")
    fig_phase.update_xaxes(title_text="time (s)")
    _save(fig_phase, "phase_timeline.png", width=1100, height=450)

    # Metrics table
    gt = metrics.get("grasp_time_s")
    rows_k = ["Overshoot","Settling time","Steady-state error","Grasp event"]
    ov = metrics.get("overshoot_pct")
    rows_v = [
        f"{ov:.1f} %" if ov is not None else "—",
        f"{metrics.get('settling_time_s', 0):.3f} s",
        f"{metrics.get('steady_state_error_m', 0)*100:.2f} cm",
        f"t = {gt:.3f} s" if gt else "not achieved",
    ]
    fig_tbl = go.Figure(data=[go.Table(
        header=dict(values=["Metric","Value"], fill_color="#3C3489",
                    font=dict(color="white", size=12), align="left"),
        cells=dict(values=[rows_k, rows_v], fill_color="#f3f3f3",
                   font=dict(color="#222", size=12), align="left")
    )])
    fig_tbl.update_layout(title="Performance metrics")
    _save(fig_tbl, "metrics_table.png", width=900, height=450)


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running simulation...")
    logger, grasp_t = run_simulation()
    n = len(logger.data["t"])
    print(f"Done — {n} steps  ({n*DT:.3f} s simulated)")
    plot_results(logger, grasp_t)
