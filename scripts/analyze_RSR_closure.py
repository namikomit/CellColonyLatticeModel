"""
Analysis script for RSR stripe closure experiment

Analyzes how sensitive stripe closure depends on initial width w.
Tests the hypothesis: W_c ~ sqrt(D / (gamma * rho_R))
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Path to simulation data
data_dir = Path(r"/nbi/nbicmplx/cell/mitarai/IndirectResistance/Brage/CellColonyLatticeModel/data/RSR_closure")  # <-- update to your local data directory

def load_simulation_data(data_dir, w):
    """Load data for a specific sensitive width."""
    # Find matching file
    pattern = f"RSR_stripe_w{w}_*.npy"
    files = list(data_dir.glob(f"RSR_stripe_w{w}_*_r_at_t.npy"))
    
    if not files:
        return None
    
    # Get run name (remove suffix)
    run_name = files[0].stem.replace('_r_at_t', '')
    
    # Load all relevant data
    data = {
        'run_info': np.load(data_dir / f"{run_name}_run_info.npy", allow_pickle=True),
        'r_at_t': np.load(data_dir / f"{run_name}_r_at_t.npy"),
        's_at_t': np.load(data_dir / f"{run_name}_s_at_t.npy"),
        'sensitive_width_at_t': np.load(data_dir / f"{run_name}_sensitive_width_at_t.npy"),
        'sensitive_center_density_at_t': np.load(data_dir / f"{run_name}_sensitive_center_density_at_t.npy"),
    }
    
    return data

def calculate_closure_metrics(data, dt=0.5):
    """
    Calculate metrics for stripe closure.
    
    Returns:
        - closure_time: Time when stripe first closes (hours), or None if never closes
        - closure_rate: Average rate of width decrease (μm/hour)
        - final_width: Width at end of simulation (μm)
        - sustained_closure: Whether closure is sustained (bool)
    """
    
    sensitive_width = data['sensitive_width_at_t']
    time_hours = np.arange(len(sensitive_width)) * dt / 60
    
    # Find when width drops below threshold (e.g., 10% of initial)
    initial_width = sensitive_width[0]
    threshold = 0.1 * initial_width
    
    closure_idx = np.where(sensitive_width < threshold)[0]
    
    if len(closure_idx) > 0:
        closure_time = time_hours[closure_idx[0]]
    else:
        closure_time = None
    
    # Calculate closure rate (linear fit to first half of simulation)
    half_point = len(sensitive_width) // 2
    if np.any(sensitive_width[:half_point] > 0):
        valid_idx = np.where(sensitive_width[:half_point] > 0)[0]
        if len(valid_idx) > 10:
            coeffs = np.polyfit(time_hours[valid_idx], sensitive_width[valid_idx], 1)
            closure_rate = -coeffs[0]  # Negative slope = closing
        else:
            closure_rate = 0
    else:
        closure_rate = 0
    
    # Final width
    final_width = sensitive_width[-1] if sensitive_width[-1] > 0 else 0
    
    # Check if closure is sustained (width stays below 50% initial for last 25% of sim)
    last_quarter = len(sensitive_width) * 3 // 4
    sustained_closure = np.all(sensitive_width[last_quarter:] < 0.5 * initial_width)
    
    return {
        'closure_time': closure_time,
        'closure_rate': closure_rate,
        'final_width': final_width,
        'sustained_closure': sustained_closure,
        'initial_width': initial_width
    }

def plot_closure_dynamics(widths_tested, dt=0.5):
    """Plot closure dynamics for all tested widths."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(widths_tested)))
    
    metrics_summary = []
    
    for i, w in enumerate(widths_tested):
        data = load_simulation_data(data_dir, w)
        
        if data is None:
            print(f"No data found for w={w}")
            continue
        
        metrics = calculate_closure_metrics(data, dt)
        metrics['w'] = w
        metrics_summary.append(metrics)
        
        time_hours = np.arange(len(data['sensitive_width_at_t'])) * dt / 60
        
        # Plot 1: Sensitive width over time
        axes[0, 0].plot(time_hours, data['sensitive_width_at_t'], 
                       label=f'w={w} μm', color=colors[i], linewidth=2)
        
        # Plot 2: Population dynamics
        axes[0, 1].plot(time_hours, data['s_at_t'], 
                       label=f'w={w} μm', color=colors[i], linewidth=2)
        
        # Plot 3: Resistant population
        axes[1, 0].plot(time_hours, data['r_at_t'], 
                       color=colors[i], linewidth=2)
        
        # Plot 4: Center density
        axes[1, 1].plot(time_hours, data['sensitive_center_density_at_t'],
                       color=colors[i], linewidth=2)
    
    # Formatting
    axes[0, 0].set_xlabel('Time (hours)', fontsize=12)
    axes[0, 0].set_ylabel('Sensitive stripe width (μm)', fontsize=12)
    axes[0, 0].set_title('Stripe Closure Dynamics', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].set_xlabel('Time (hours)', fontsize=12)
    axes[0, 1].set_ylabel('Sensitive cell count', fontsize=12)
    axes[0, 1].set_title('Sensitive Population', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    axes[1, 0].set_xlabel('Time (hours)', fontsize=12)
    axes[1, 0].set_ylabel('Resistant cell count', fontsize=12)
    axes[1, 0].set_title('Resistant Population', fontsize=14, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    axes[1, 1].set_xlabel('Time (hours)', fontsize=12)
    axes[1, 1].set_ylabel('Center density (cells/μm)', fontsize=12)
    axes[1, 1].set_title('Sensitive Center Density', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(data_dir / 'closure_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return metrics_summary

def analyze_critical_width(metrics_summary, D=12000, gamma=0.025, rho_R=1):
    """
    Analyze critical width and compare to theoretical prediction.
    
    Args:
        metrics_summary: List of dicts with closure metrics for each width
        D: Diffusion coefficient (μm²/min)
        gamma: Degradation rate (1/min for full diffusion cell)
        rho_R: Resistant cell density (cells/μm²)
    """
    
    # Theoretical critical width
    lambda_theory = np.sqrt(D / gamma)  # Characteristic length
    W_c_theory = 2 * lambda_theory  # Critical width (rough estimate)
    
    print("\n" + "="*60)
    print("CRITICAL WIDTH ANALYSIS")
    print("="*60)
    print(f"\nTheoretical prediction:")
    print(f"  λ = √(D/γ) = √({D}/{gamma}) = {lambda_theory:.1f} μm")
    print(f"  W_c ≈ 2λ = {W_c_theory:.1f} μm")
    print("\n" + "-"*60)
    print(f"{'Width (μm)':<15} {'Closure Rate':<15} {'Final Width':<15} {'Closed?'}")
    print("-"*60)
    
    widths = []
    closure_rates = []
    final_widths = []
    closures = []
    
    for m in sorted(metrics_summary, key=lambda x: x['w']):
        widths.append(m['w'])
        closure_rates.append(m['closure_rate'])
        final_widths.append(m['final_width'])
        closures.append(m['sustained_closure'])
        
        print(f"{m['w']:<15} {m['closure_rate']:<15.2f} {m['final_width']:<15.1f} "
              f"{'Yes' if m['sustained_closure'] else 'No'}")
    
    print("-"*60)
    
    # Plot closure rate vs width
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Closure rate vs width
    axes[0].plot(widths, closure_rates, 'o-', markersize=10, linewidth=2)
    axes[0].axvline(W_c_theory, color='red', linestyle='--', linewidth=2, 
                   label=f'W_c (theory) = {W_c_theory:.0f} μm')
    axes[0].axvline(lambda_theory, color='orange', linestyle='--', linewidth=2, 
                   label=f'λ = {lambda_theory:.0f} μm')
    axes[0].set_xlabel('Initial sensitive width (μm)', fontsize=12)
    axes[0].set_ylabel('Closure rate (μm/hour)', fontsize=12)
    axes[0].set_title('Closure Rate vs Initial Width', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # Plot 2: Final width vs initial width
    axes[1].plot(widths, final_widths, 'o-', markersize=10, linewidth=2, label='Final width')
    axes[1].plot([min(widths), max(widths)], [min(widths), max(widths)], 
                'k--', alpha=0.3, label='No closure (1:1 line)')
    axes[1].axvline(W_c_theory, color='red', linestyle='--', linewidth=2, 
                   label=f'W_c (theory) = {W_c_theory:.0f} μm')
    axes[1].set_xlabel('Initial sensitive width (μm)', fontsize=12)
    axes[1].set_ylabel('Final sensitive width (μm)', fontsize=12)
    axes[1].set_title('Final vs Initial Width', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(data_dir / 'critical_width_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Estimate empirical critical width
    # Find where closure rate drops significantly
    if len(closure_rates) > 1:
        rates_normalized = np.array(closure_rates) / max(closure_rates)
        
        # Critical width = where rate drops below 50% of maximum
        critical_idx = np.where(rates_normalized < 0.5)[0]
        if len(critical_idx) > 0:
            W_c_empirical = widths[critical_idx[0]]
            print(f"\nEmpirical critical width: W_c ≈ {W_c_empirical} μm")
            print(f"Ratio (empirical/theory): {W_c_empirical/W_c_theory:.2f}")
        else:
            print("\nAll tested widths show significant closure.")
            print("Need to test smaller widths to find W_c.")
    
    return {
        'lambda_theory': lambda_theory,
        'W_c_theory': W_c_theory,
        'widths': widths,
        'closure_rates': closure_rates,
        'final_widths': final_widths
    }

if __name__ == "__main__":
    
    # Widths tested in simulation
    widths_tested = [200, 400, 600, 800, 1000, 1200, 1500, 2000]
    
    print("Loading and analyzing simulation data...")
    
    # Plot dynamics
    metrics = plot_closure_dynamics(widths_tested, dt=0.5)
    
    # Analyze critical width
    if len(metrics) > 0:
        # Extract parameters from first simulation
        data_sample = load_simulation_data(data_dir, widths_tested[0])
        run_info = dict(data_sample['run_info'])
        
        D = float(run_info['D'])
        gamma = float(run_info['gamma']) * float(run_info['ddx'])**2  # Convert back to non-scaled
        
        analysis = analyze_critical_width(metrics, D=D, gamma=gamma)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Plots saved to: {data_dir}")
    else:
        print("\nNo simulation data found!")
        print(f"Looking in: {data_dir}")
        print("Make sure to run the simulations first.")
