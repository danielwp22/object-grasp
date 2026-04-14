# Controller Comparison: PID vs PD + Feedforward

## Summary

This document compares two controller architectures for the moving object grasp task:
1. **PID + Feedforward** (`grasp_controller.py`)
2. **PD + Feedforward** (`grasp_controller_pd.py`)

## Performance Results

| Metric | PID + FF | PD + FF (optimized) | Winner |
|--------|----------|---------------------|--------|
| **Grasp time** | 0.84s | 1.025s | **PID** (18% faster) |
| **Overshoot** | 242.6% | 221.3% | **PD** (9% lower) |
| **Settling time** | 0.15s | 0.235s | **PID** (36% faster) |
| **Steady-state error** | 4.74cm | 5.83cm | **PID** (19% better) |
| **Overall** | - | - | **PID wins 3/4 metrics** |

## Controller Configurations

### PID + Feedforward (`grasp_controller.py`)
```python
Kp = 180.0    # Proportional gain
Ki = 40.0     # Integral gain (eliminates steady-state error)
Kd = 15.0     # Derivative gain (damping)
ff_gain = 0.7 # Reduced to minimize chattering
```

**Architecture:**
```
u = u_ff + u_fb
u_ff = 0.7 * (m*a_des + b*v_des)  # Feedforward (model inversion)
u_fb = Kp*e + Ki*∫e + Kd*(de/dt)  # Feedback (error correction)
```

**Key features:**
- Integral term eliminates steady-state error from model mismatch
- Lower feedforward gain (0.7) reduces sensitivity to model errors
- Moderate Kp/Kd for stable tracking without excessive aggressiveness

### PD + Feedforward (`grasp_controller_pd.py`)
```python
Kp = 400.0    # Proportional gain (tuned via pole placement)
Ki = 0.0      # Disabled
Kd = 35.0     # Derivative gain (tuned via pole placement)
ff_gain = 1.0 # Full feedforward compensation
```

**Pole placement design:**
- Target: ζ=1.0 (critically damped), ω_n=20 rad/s
- Predicted: 7.3% overshoot, 65ms settling time, 80° phase margin
- Actual: 221% overshoot, 235ms settling time (theory vs practice gap!)

**Why the discrepancy?**
1. Pole placement assumes **ideal step response** without feedforward
2. Full feedforward (ff_gain=1.0) adds significant control effort
3. Velocity saturation (3.0 m/s limit) creates nonlinearity
4. Moving target during APPROACH phase ≠ step input
5. Without integral term, model errors accumulate as steady-state error

## Advanced Tuning Analysis

See `tune_pd_advanced.py` for pole placement and optimization-based tuning methods.

### Tuning Methods Explored:
1. **Critically damped** (ζ=1.0): Minimal overshoot, slow response
2. **Butterworth** (ζ=0.707): Balanced, ~13% overshoot
3. **Fast response** (ζ=0.8, high ω_n): Aggressive, requires careful tuning
4. **Optimization-based**: Minimize settling time with overshoot constraint

### Key Insights from Analysis:

**Step response metrics (ideal plant, no feedforward):**
- Kp=400, Kd=35 → 7.3% overshoot, 65ms settling, 80° PM ✓ theory
- Kp=180, Kd=15 → 10.3% overshoot, 115ms settling, 71° PM

**Actual grasp task (with feedforward + saturation):**
- Kp=400, Kd=35 → 221% overshoot, 235ms settling ✗ much worse!
- Kp=180, Kd=15, Ki=40 → 243% overshoot, 150ms settling ✓ better

**Conclusion:** Classical tuning methods (Bode, pole placement) predict ideal linear system behavior, but real tasks have:
- Feedforward compensation (not modeled in classical analysis)
- Actuator saturation (nonlinear)
- Moving targets (time-varying setpoints)
- Model mismatch (requires integral term)

## Chattering Analysis

Both controllers exhibit velocity chattering (oscillations around 3.0 m/s limit).

**Root causes:**
1. KF velocity estimates have noise (even though they're optimal)
2. Kd term amplifies velocity noise
3. Velocity saturation creates bang-bang effect
4. Physics-based acceleration helps, but velocity is still noisy

**Mitigation strategies tried:**
- ✓ Use physics-based ball acceleration (smooth, known)
- ✓ Reduce feedforward gain to 0.7 (less model sensitivity)
- ✓ Reduce Kd from 28 to 15 (less noise amplification)
- ✗ Removing integral term (made steady-state error worse)
- ✗ Increasing Kd to 35 (made chattering worse)

**Remaining options:**
- Low-pass filter on KF velocity before control
- Reduce Kd further (but hurts damping)
- Accept chattering as fundamental limit given sensor noise + velocity limit

## Recommendations

### For this grasp task:
**Use PID + Feedforward** (`grasp_controller.py`)
- Faster grasp time (0.84s vs 1.025s)
- Lower steady-state error (4.74cm vs 5.83cm)
- Faster settling (0.15s vs 0.235s)
- Integral term compensates for model errors

### For other applications:
- **PD + FF**: Good when plant model is very accurate and no steady-state precision required
- **PID + FF**: Better for real systems with model uncertainty, disturbances, or precision requirements
- **Pure PD**: Only if feedforward unavailable and can tolerate steady-state error

## Files

| File | Description |
|------|-------------|
| `grasp_controller.py` | PID + Feedforward controller (recommended) |
| `grasp_controller_pd.py` | PD + Feedforward controller (optimized gains) |
| `tune_pd_advanced.py` | Pole placement and optimization-based PD tuning |
| `tune_pid_bode.py` | Frequency-domain PID tuning (Bode/Ziegler-Nichols) |
| `analyze_frequency_response.py` | Phase margin analysis for current gains |
| `test_step_response.py` | Step response validation of Bode predictions |
| `CONTROLLER_COMPARISON.md` | This document |

## Visualization

Pole placement analysis showing how damping ratio (ζ) and natural frequency (ω_n) affect:
- Step response shape
- Pole locations in s-plane
- Settling time vs overshoot tradeoff

See `assets/pd_tuning_analysis.png` for detailed plots.

---

**Bottom line:** For robotic manipulation with model uncertainty, **PID + Feedforward** provides better performance than pure PD + Feedforward, even when PD gains are optimally tuned. The integral term is valuable for handling real-world imperfections.
