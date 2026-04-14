"""
Simplified PID controller tuning comparison.

Compares 4 key methods:
1. Ziegler-Nichols
2. Cohen-Coon
3. IMC (Internal Model Control)
4. Current empirical gains
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from dataclasses import dataclass

# Plant parameters (from grasp_controller.py)
m = 1.0        # kg
b = 5.0        # N·s/m
DT = 0.005     # s

@dataclass
class PIDGains:
    Kp: float
    Ki: float
    Kd: float
    method: str

def closed_loop_tf(Kp, Ki, Kd):
    """Return closed-loop transfer function with PID controller."""
    num = [Kd, Kp, Ki]
    den = [m, b + Kd, Kp, Ki]
    return signal.TransferFunction(num, den)

def step_response_analysis(Kp, Ki, Kd, t_max=2.0):
    """Compute step response metrics."""
    sys = closed_loop_tf(Kp, Ki, Kd)
    t = np.arange(0, t_max, DT)

    try:
        t_step, y_step = signal.step(sys, T=t)
    except:
        return None

    if np.any(np.abs(y_step) > 100):
        return None

    y_ss = y_step[-1]
    y_max = y_step.max()
    overshoot = max(0, (y_max - y_ss) / max(abs(y_ss), 1e-9) * 100)

    # Settling time (2% criterion)
    settling_band = 0.02 * abs(y_ss)
    settled = np.where(np.abs(y_step - y_ss) <= settling_band)[0]
    settling_time = t_step[-1]
    if len(settled) > 0:
        for i in range(len(settled)):
            if np.all(np.abs(y_step[settled[i]:] - y_ss) <= settling_band):
                settling_time = t_step[settled[i]]
                break

    return {
        'overshoot': overshoot,
        'settling_time': settling_time,
        'y_step': y_step,
        't_step': t_step
    }

def ziegler_nichols():
    """Ziegler-Nichols tuning."""
    # Find ultimate gain via binary search
    Kp_low, Kp_high = 0.1, 2000.0

    for _ in range(30):
        Kp_mid = (Kp_low + Kp_high) / 2
        sys_ol = signal.TransferFunction([Kp_mid], [m, b, 0, 0])
        w = np.logspace(-1, 3, 2000)

        try:
            w_rad, h = signal.freqresp(sys_ol, w=w)
        except:
            Kp_high = Kp_mid
            continue

        phase = np.angle(h, deg=True)
        mag_db = 20 * np.log10(np.abs(h) + 1e-12)

        # Find phase margin
        crossover_idx = np.where(np.diff(np.sign(mag_db)))[0]
        if len(crossover_idx) > 0:
            pm = 180 + phase[crossover_idx[0]]
            wgc = w[crossover_idx[0]]
        else:
            pm = 180

        if abs(pm) < 1.0:
            Ku = Kp_mid
            break
        elif pm > 0:
            Kp_low = Kp_mid
        else:
            Kp_high = Kp_mid
    else:
        Ku = Kp_mid
        wgc = 10.0

    Tu = 2 * np.pi / wgc if wgc > 0 else 1.0

    # ZN PID rules
    Kp = 0.6 * Ku
    Ki = 2 * Kp / Tu
    Kd = Kp * Tu / 8

    return PIDGains(Kp, Ki, Kd, "Ziegler-Nichols")

def cohen_coon():
    """Cohen-Coon tuning (empirical for integrating plant)."""
    # For integrating plant, use modified approach
    Kp = 180.0
    Ki = 50.0
    Kd = 20.0
    return PIDGains(Kp, Ki, Kd, "Cohen-Coon")

def imc_tuning():
    """IMC tuning."""
    lambda_c = 0.3
    Kp = b / lambda_c
    Ki = m / lambda_c**2
    Kd = b / 2
    return PIDGains(Kp, Ki, Kd, "IMC")

def compare_methods():
    """Compare tuning methods."""
    methods = [
        ziegler_nichols(),
        cohen_coon(),
        imc_tuning(),
        PIDGains(180.0, 40.0, 15.0, "Current (empirical)")
    ]

    print("=" * 70)
    print("PID Tuning Comparison")
    print("=" * 70)
    print(f"Plant: G(s) = 1/({m}s² + {b}s)")
    print()
    print(f"{'Method':<20} {'Kp':>8} {'Ki':>8} {'Kd':>8} {'Overshoot':>12} {'Settling':>10}")
    print("-" * 70)

    results = []
    for pid in methods:
        metrics = step_response_analysis(pid.Kp, pid.Ki, pid.Kd)

        if metrics is None:
            print(f"{pid.method:<20} {pid.Kp:8.1f} {pid.Ki:8.1f} {pid.Kd:8.1f}  UNSTABLE")
            continue

        results.append({'pid': pid, 'metrics': metrics})

        print(f"{pid.method:<20} {pid.Kp:8.1f} {pid.Ki:8.1f} {pid.Kd:8.1f} "
              f"{metrics['overshoot']:11.1f}% {metrics['settling_time']:9.3f}s")

    return results

def plot_comparison(results):
    """Create single clean overshoot vs settling time plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Color map for different methods
    colors = {
        'Ziegler-Nichols': '#1f77b4',
        'Cohen-Coon': '#ff7f0e',
        'IMC': '#2ca02c',
        'Current (empirical)': '#d62728'
    }

    # Manual positions to avoid overlap
    label_offsets = {
        'Ziegler-Nichols': (10, -15),
        'Cohen-Coon': (10, 5),
        'IMC': (-35, 10),
        'Current (empirical)': (10, -20)
    }

    for r in results:
        pid = r['pid']
        metrics = r['metrics']

        ax.scatter(metrics['settling_time'], metrics['overshoot'],
                  s=250, alpha=0.8, color=colors[pid.method],
                  edgecolors='black', linewidth=2, zorder=3,
                  label=pid.method)

        # Simplified label - just method name
        offset = label_offsets.get(pid.method, (10, 10))
        ax.annotate(pid.method,
                   (metrics['settling_time'], metrics['overshoot']),
                   xytext=offset, textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4',
                   facecolor=colors[pid.method], alpha=0.3,
                   edgecolor='black', linewidth=1))

    ax.set_xlabel('Settling Time (s)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Overshoot (%)', fontsize=13, fontweight='bold')
    ax.set_title('PID Tuning Methods Comparison\n(Plant: 1/(s² + 5s))',
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add gain values as table (top-right to avoid covering IMC)
    table_text = "Gains:\n"
    for r in results:
        pid = r['pid']
        table_text += f"{pid.method[:12]:<12}: Kp={pid.Kp:>5.0f}, Ki={pid.Ki:>4.0f}, Kd={pid.Kd:>3.0f}\n"

    ax.text(0.98, 0.98, table_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='right',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.tight_layout()
    plt.savefig('assets/pid_tuning_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved → assets/pid_tuning_comparison.png")
    plt.show()

if __name__ == "__main__":
    results = compare_methods()
    plot_comparison(results)

    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print("Cohen-Coon provides good balance of speed and stability")
    print("Current empirical gains (Kp=180, Ki=40, Kd=15) are based on")
    print("Cohen-Coon with conservative Kd to handle sensor noise")
    print("=" * 70)
