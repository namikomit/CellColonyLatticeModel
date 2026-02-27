"""
Well-Mixed Co-Culture Model: Resistant vs Sensitive Bacteria with Antibiotics

This model simulates competition between resistant (R) and sensitive (S) bacteria
in a well-mixed environment (1 mL liquid culture) with antibiotics (A) and nutrients (N).

ODE System:
-----------
dR/dt = λ_r * f_N(N) * R
dS/dt = λ_s * f_N(N) * f_A(A) * S
dA/dt = -γ * R * A
dN/dt = -(dR/dt + dS/dt)    [One cell division consumes one unit of nutrient]

Where:
- f_N(N) = N/(k_N + N)                : Monod nutrient limitation
- f_A(A) = 1/(1 + (A/k_A)³)      for A<4, linear interpolation to 0     : Hill antibiotic inhibition
- γ : antibiotic degradation rate by R cells (NO background decay)
- K : carrying capacity (max cells in 1 mL)

User specifications:
- R₀ = S₀ = 10,000 cells
- N₀ = 10⁹ nutrient units → Final cells ≈ 10⁹
- K = 5×10⁸ cells (Monod constant for logistic growth)
- Volume = 1 mL
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================================
# MODEL PARAMETERS (matching spatial simulation)
# ============================================================================

class Parameters:
    """Container for model parameters."""
    
    def __init__(self):
        # Growth rates (from spatial model)
        self.lambda_r = 1.93  # h⁻¹, resistant growth rate
        self.lambda_s = 1.93  # h⁻¹, sensitive growth rate
        
        # Convert to per-minute for consistency with spatial code
        self.lambda_r_min = self.lambda_r / (np.log(2) * 60)  # min⁻¹
        self.lambda_s_min = self.lambda_s / (np.log(2) * 60)  # min⁻¹
        
        # Antibiotic parameters (from spatial model)
        self.k_A = 2.35       # μg/mL, antibiotic IC50
        self.gamma = 2.78e-5  # min⁻¹, degradation rate per R cell
        
        # Nutrient parameters
        self.k_N = 5e8       # Nutrient Monod constant cells/mL
                              # Set to 0 for no nutrient limitation (saturating)
        
        
        # Volume
        self.volume = 1.0     # mL
    
    def __repr__(self):
        return (f"Parameters:\n"
                f"  λ_r = {self.lambda_r:.2f} h⁻¹\n"
                f"  λ_s = {self.lambda_s:.2f} h⁻¹\n"
                f"  k_A = {self.k_A:.2f} μg/mL\n"
                f"  γ = {self.gamma:.2e} min⁻¹\n"
                f"  k_N = {self.k_N:.2f}\n"
                f"  Volume = {self.volume} mL\n")

# ============================================================================
# ODE SYSTEM
# ============================================================================

def ode_system(t, y, params):
    """
    ODE system for well-mixed R-S competition.
    
    Parameters:
    -----------
    t : float
        Time (minutes)
    y : array [R, S, A, N]
        State variables:
        - R: resistant cell count
        - S: sensitive cell count
        - A: antibiotic concentration (μg/mL)
        - N: nutrient concentration (units)
    params : Parameters
        Model parameters
    
    Returns:
    --------
    dydt : array [dR/dt, dS/dt, dA/dt, dN/dt]
    """
    R, S, A, N = y
    
    # Prevent negative values
    R = max(R, 0)
    S = max(S, 0)
    A = max(A, 0)
    N = max(N, 0)
    
    # Nutrient limitation term (Monod kinetics)
    if params.k_N > 0:
        f_N = N / (params.k_N + N)
    else:
        f_N = 1.0  # No nutrient limitation (saturating)
    
    # Antibiotic inhibition term (Hill function with n=3)
    if A > 0 and A < 4:
        f_A = 1.0 / (1.0 + (A / params.k_A)**3)
    elif A >=4 and A < 5: 
        f_A = 1.0 / (1.0 + (4 / params.k_A)**3)*(5-A)  # Linear interpolation to 0 between 4 and 5 μg/mL
    elif A >= 5:
        f_A = 0.0  # Complete inhibition above 5 μg/mL
    else:
        f_A = 1.0  # No antibiotic inhibition

    # Growth rates
    growth_R = params.lambda_r_min * f_N
    growth_S = params.lambda_s_min * f_N * f_A 
    
    # ODEs
    dR_dt = growth_R * R
    dS_dt = growth_S * S
    dA_dt = -params.gamma * R * A  # Only degradation by R cells, no background decay
    dN_dt = -(dR_dt + dS_dt)       # One cell division consumes one nutrient unit
    
    return [dR_dt, dS_dt, dA_dt, dN_dt]

# ============================================================================
# SIMULATION FUNCTION
# ============================================================================

def run_simulation(R0, S0, A0, N0, t_max, params):
    """
    Run a single simulation.
    
    Parameters:
    -----------
    R0, S0 : float
        Initial cell counts
    A0 : float
        Initial antibiotic concentration (μg/mL)
    N0 : float
        Initial nutrient concentration
    t_max : float
        Simulation time (hours)
    params : Parameters
        Model parameters
    
    Returns:
    --------
    solution : OdeResult
        Solution object from solve_ivp
    """
    y0 = [R0, S0, A0, N0]
    t_span = (0, t_max * 60)  # Convert hours to minutes
    t_eval = np.linspace(0, t_max * 60, 1000)  # Evaluate at 1000 points
    
    # Solve ODE
    sol = solve_ivp(
        ode_system,
        t_span,
        y0,
        args=(params,),
        method='LSODA',  # Good for stiff ODEs
        t_eval=t_eval,
        dense_output=True
    )
    
    return sol

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_timeseries(sol, params, title="", save_path=None):
    """Plot time series of R, S, A, N."""
    
    t_hours = sol.t / 60  # Convert to hours
    R, S, A, N = sol.y
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Cell populations (linear scale)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_hours, R, 'b-', linewidth=2, label='Resistant (R)')
    ax1.plot(t_hours, S, 'y-', linewidth=2, label='Sensitive (S)')
    ax1.plot(t_hours, R + S, 'k--', linewidth=1.5, label='Total', alpha=0.7)
    ax1.set_xlabel('Time (hours)', fontsize=11)
    ax1.set_ylabel('Cell count', fontsize=11)
    ax1.set_title('Population Dynamics', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Panel 2: Cell populations (log scale)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(t_hours, R + 1, 'b-', linewidth=2, label='Resistant (R)')
    ax2.semilogy(t_hours, S + 1, 'y-', linewidth=2, label='Sensitive (S)')
    ax2.set_xlabel('Time (hours)', fontsize=11)
    ax2.set_ylabel('Cell count (log scale)', fontsize=11)
    ax2.set_title('Population Dynamics (Log Scale)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # Panel 3: R/S ratio over time
    ax3 = fig.add_subplot(gs[1, 0])
    ratio = R / (S + 1e-10)  # Avoid division by zero
    ax3.semilogy(t_hours, ratio, 'g-', linewidth=2)
    ax3.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='R = S')
    ax3.set_xlabel('Time (hours)', fontsize=11)
    ax3.set_ylabel('R/S ratio (log scale)', fontsize=11)
    ax3.set_title('Competitive Outcome', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
    # Panel 4: Antibiotic concentration
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t_hours, A, 'r-', linewidth=2)
    ax4.axhline(y=params.k_A, color='k', linestyle='--', alpha=0.5, 
                label=f'k_A = {params.k_A:.2f} μg/mL')
    ax4.set_xlabel('Time (hours)', fontsize=11)
    ax4.set_ylabel('Antibiotic (μg/mL)', fontsize=11)
    ax4.set_title('Antibiotic Dynamics', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    
    # Panel 5: Nutrient concentration
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(t_hours, N, 'm-', linewidth=2)
    ax5.set_xlabel('Time (hours)', fontsize=11)
    ax5.set_ylabel('Nutrient (a.u.)', fontsize=11)
    ax5.set_title('Nutrient Dynamics', fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3)
    
    # Panel 6: Summary statistics
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    # Calculate final values
    R_final = R[-1]
    S_final = S[-1]
    A_final = A[-1]
    N_final = N[-1]
    ratio_final = R_final / (S_final + 1e-10)
    
    summary = f"Initial Conditions:\n"
    summary += f"  R₀ = {R[0]:.2e}\n"
    summary += f"  S₀ = {S[0]:.2e}\n"
    summary += f"  A₀ = {A[0]:.2f} μg/mL\n"
    summary += f"  N₀ = {N[0]:.2f}\n\n"
    summary += f"Final Values (t = {t_hours[-1]:.1f} h):\n"
    summary += f"  R = {R_final:.2e}\n"
    summary += f"  S = {S_final:.2e}\n"
    summary += f"  A = {A_final:.4f} μg/mL\n"
    summary += f"  N = {N_final:.4f}\n\n"
    summary += f"Outcome:\n"
    summary += f"  R/S ratio = {ratio_final:.2e}\n"
    if ratio_final > 10:
        summary += f"  → R dominates\n"
    elif ratio_final < 0.1:
        summary += f"  → S dominates\n"
    else:
        summary += f"  → Coexistence\n"
    
    ax6.text(0.1, 0.95, summary, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def plot_parameter_scan(R0, S0, A0_values, N0, t_max, params, save_path=None):
    """
    Scan over initial antibiotic concentrations and plot final outcomes.
    """
    
    R_finals = []
    S_finals = []
    ratios = []
    
    print(f"\nScanning A₀ from {min(A0_values):.1f} to {max(A0_values):.1f} μg/mL...")
    
    for A0 in A0_values:
        sol = run_simulation(R0, S0, A0, N0, t_max, params)
        R_final = sol.y[0, -1]
        S_final = sol.y[1, -1]
        
        R_finals.append(R_final)
        S_finals.append(S_final)
        ratios.append(R_final / (S_final + 1e-10))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Final populations
    ax1 = axes[0]
    ax1.semilogy(A0_values, R_finals, 'bo-', linewidth=2, markersize=8, label='R')
    ax1.semilogy(A0_values, S_finals, 'yo-', linewidth=2, markersize=8, label='S')
    ax1.axvline(x=params.k_A, color='r', linestyle='--', alpha=0.5, 
                label=f'k_A = {params.k_A:.2f}')
    ax1.set_xlabel('Initial Antibiotic A₀ (μg/mL)', fontsize=12)
    ax1.set_ylabel('Final Population (log scale)', fontsize=12)
    ax1.set_title('Final Yield vs Initial Antibiotic', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Panel 2: R/S ratio
    ax2 = axes[1]
    ax2.semilogy(A0_values, ratios, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='R = S')
    ax2.axvline(x=params.k_A, color='r', linestyle='--', alpha=0.5, 
                label=f'k_A = {params.k_A:.2f}')
    ax2.set_xlabel('Initial Antibiotic A₀ (μg/mL)', fontsize=12)
    ax2.set_ylabel('Final R/S Ratio (log scale)', fontsize=12)
    ax2.set_title('Competitive Outcome vs Initial Antibiotic', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def plot_phase_diagram(R0, S0, A0_values, N0, t_max, params, save_path=None):
    """
    Create comprehensive phase diagram showing competitive outcome.
    """
    
    R_finals = []
    S_finals = []
    A_finals = []
    N_finals = []
    ratios = []
    R_fractions = []
    
    print(f"\nScanning A₀ from {min(A0_values):.1f} to {max(A0_values):.1f} μg/mL...")
    
    for i, A0 in enumerate(A0_values):
        print(f"  Progress: {i+1}/{len(A0_values)} - A₀ = {A0:.2f} μg/mL...", end='\r')
        sol = run_simulation(R0, S0, A0, N0, t_max, params)
        
        R_final = sol.y[0, -1]
        S_final = sol.y[1, -1]
        A_final = sol.y[2, -1]
        N_final = sol.y[3, -1]
        
        R_finals.append(R_final)
        S_finals.append(S_final)
        A_finals.append(A_final)
        N_finals.append(N_final)
        ratios.append(R_final / (S_final + 1e-10))
        R_fractions.append(R_final / (R_final + S_final + 1e-10))
    
    print()  # New line after progress
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # -------------------------------------------------------------------------
    # Panel 1: Final populations (log scale)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(A0_values, R_finals, 'bo-', linewidth=2, markersize=6, label='R (Resistant)', alpha=0.8)
    ax1.semilogy(A0_values, S_finals, 'yo-', linewidth=2, markersize=6, label='S (Sensitive)', alpha=0.8)
    ax1.semilogy(A0_values, np.array(R_finals) + np.array(S_finals), 'ko--', 
                linewidth=1.5, markersize=4, label='Total', alpha=0.5)
    ax1.axvline(x=params.k_A, color='r', linestyle='--', linewidth=2, alpha=0.7,
                label=f'k_A = {params.k_A:.2f}')
    ax1.axhline(y=R0, color='b', linestyle=':', alpha=0.3)
    ax1.axhline(y=S0, color='y', linestyle=':', alpha=0.3)
    ax1.set_xlabel('Initial Antibiotic A₀ (μg/mL)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Final Population (cells)', fontsize=12, fontweight='bold')
    ax1.set_title('Final Yield vs Initial Antibiotic', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Panel 2: R/S ratio (log scale)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(A0_values, ratios, 'go-', linewidth=2.5, markersize=7)
    ax2.axhline(y=1, color='k', linestyle='--', linewidth=2, alpha=0.7, label='R = S')
    ax2.axvline(x=params.k_A, color='r', linestyle='--', linewidth=2, alpha=0.7,
                label=f'k_A = {params.k_A:.2f}')
    
    # Shade regions
    ax2.axhspan(0.1, 10, alpha=0.1, color='gray', label='Coexistence zone')
    
    ax2.set_xlabel('Initial Antibiotic A₀ (μg/mL)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Final R/S Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Competitive Outcome (Phase Diagram)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(alpha=0.3)
    ax2.set_ylim([min(ratios)*0.5, max(ratios)*2])
    
    # -------------------------------------------------------------------------
    # Panel 3: R fraction
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(A0_values, R_fractions, 'mo-', linewidth=2.5, markersize=7)
    ax3.axhline(y=0.5, color='k', linestyle='--', linewidth=2, alpha=0.7, label='Equal (50%)')
    ax3.axvline(x=params.k_A, color='r', linestyle='--', linewidth=2, alpha=0.7,
                label=f'k_A = {params.k_A:.2f}')
    ax3.set_xlabel('Initial Antibiotic A₀ (μg/mL)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('R Fraction (R/(R+S))', fontsize=12, fontweight='bold')
    ax3.set_title('Resistant Fraction in Final Population', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # -------------------------------------------------------------------------
    # Panel 4: Final antibiotic
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(A0_values, A_finals, 'ro-', linewidth=2, markersize=6)
    ax4.axvline(x=params.k_A, color='r', linestyle='--', linewidth=2, alpha=0.7,
                label=f'k_A = {params.k_A:.2f}')
    ax4.set_xlabel('Initial Antibiotic A₀ (μg/mL)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Final Antibiotic (μg/mL)', fontsize=12, fontweight='bold')
    ax4.set_title('Antibiotic Remaining at End', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10, loc='best')
    ax4.grid(alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Panel 5: Final nutrients
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.semilogy(A0_values, np.array(N_finals) + 1, 'co-', linewidth=2, markersize=6)
    ax5.axvline(x=params.k_A, color='r', linestyle='--', linewidth=2, alpha=0.7,
                label=f'k_A = {params.k_A:.2f}')
    ax5.set_xlabel('Initial Antibiotic A₀ (μg/mL)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Final Nutrients (log scale)', fontsize=12, fontweight='bold')
    ax5.set_title('Nutrients Remaining at End', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10, loc='best')
    ax5.grid(alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Panel 6: Classification table
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Classify outcomes
    n_R_dominates = sum(1 for r in ratios if r > 10)
    n_S_dominates = sum(1 for r in ratios if r < 0.1)
    n_coexist = len(ratios) - n_R_dominates - n_S_dominates
    
    # Find transition points
    transition_A0 = []
    for i in range(len(ratios) - 1):
        if (ratios[i] < 1 and ratios[i+1] > 1) or (ratios[i] > 1 and ratios[i+1] < 1):
            transition_A0.append((A0_values[i] + A0_values[i+1]) / 2)
    
    summary = "Outcome Classification:\n"
    summary += "="*40 + "\n\n"
    summary += f"R dominates (R/S > 10):\n"
    summary += f"  {n_R_dominates}/{len(A0_values)} cases ({100*n_R_dominates/len(A0_values):.1f}%)\n\n"
    summary += f"S dominates (R/S < 0.1):\n"
    summary += f"  {n_S_dominates}/{len(A0_values)} cases ({100*n_S_dominates/len(A0_values):.1f}%)\n\n"
    summary += f"Coexistence (0.1 ≤ R/S ≤ 10):\n"
    summary += f"  {n_coexist}/{len(A0_values)} cases ({100*n_coexist/len(A0_values):.1f}%)\n\n"
    summary += "="*40 + "\n\n"
    summary += f"k_A = {params.k_A:.2f} μg/mL\n\n"
    
    if transition_A0:
        summary += f"Transition point(s):\n"
        for ta in transition_A0:
            summary += f"  A₀ ≈ {ta:.2f} μg/mL\n"
    
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle(f'Phase Diagram: R₀={R0:,.0f}, S₀={S0:,.0f}, N₀={N0:.1e}, t={t_max}h', 
                fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run simulations with user specifications."""
    
    print("="*70)
    print("Well-Mixed Co-Culture Model: R vs S with Antibiotics")
    print("="*70)
    print()
    
    # Initialize parameters
    params = Parameters()
    print(params)
    print()
    
    # -------------------------------------------------------------------------
    # User specifications
    # -------------------------------------------------------------------------
    R0 = 1000       # Initial resistant cells (fixed)
    S0 = 1000       # Initial sensitive cells (fixed)
    N0 = 1e9         # Initial nutrients (→ final cells ≈ 10⁹)
    t_max = 6       # Simulation time (hours)
    
    # Antibiotic range to test
    A0_values = np.linspace(5, 10, 11)  # 0 to 10 μg/mL in steps of 0.5
    
    print(f"Fixed initial conditions:")
    print(f"  R₀ = {R0:,.0f} cells")
    print(f"  S₀ = {S0:,.0f} cells")
    print(f"  N₀ = {N0:.2e} nutrient units")
    print(f"  Simulation time = {t_max} hours")
    print()
    print(f"Scanning A₀ from {A0_values[0]:.1f} to {A0_values[-1]:.1f} μg/mL")
    print(f"  ({len(A0_values)} values)")
    print()
    
    # -------------------------------------------------------------------------
    # Example 1: Single simulation at A₀ = k_A
    # -------------------------------------------------------------------------
    print("Example 1: Single Simulation at A₀ = 5 μg/mL")
    print("-" * 70)
    
    A0_example = 5
    
    sol = run_simulation(R0, S0, A0_example, N0, t_max, params)
    
    plot_timeseries(sol, params, 
                   title=f"Well-Mixed Competition: R₀={R0:,.0f}, S₀={S0:,.0f}, A₀={A0_example:.2f} μg/mL",
                   save_path="wellmixed_single.png")
    
    # -------------------------------------------------------------------------
    # Example 2: Parameter scan over A₀
    # -------------------------------------------------------------------------
    print()
    print("Example 2: Scanning Over Initial Antibiotic A₀")
    print("-" * 70)
    
    plot_parameter_scan(R0, S0, A0_values, N0, t_max, params,
                       save_path="wellmixed_scan.png")
    
    # -------------------------------------------------------------------------
    # Example 3: Phase diagram (R/S ratio vs A₀)
    # -------------------------------------------------------------------------
    print()
    print("Example 3: Creating Phase Diagram")
    print("-" * 70)
    
    plot_phase_diagram(R0, S0, A0_values, N0, t_max, params,
                      save_path="wellmixed_phase.png")
    
    print()
    print("="*70)
    print("Simulations Complete!")
    print("="*70)
    print()
    print("Files created:")
    print("  wellmixed_single.png  - Time series for A₀ = k_A")
    print("  wellmixed_scan.png    - Final populations vs A₀")
    print("  wellmixed_phase.png   - Phase diagram (R/S ratio)")
    print()

if __name__ == "__main__":
    main()
