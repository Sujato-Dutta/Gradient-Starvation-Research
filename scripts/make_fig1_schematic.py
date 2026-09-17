import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up figure with publication typography
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(8.2, 3.4), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# Colors
c_shared = '#1565C0'
c_both = '#C62828'
c_weak = '#2E7D32'
c_diag = '#6A1B9A'
c_gate = '#E65100'

# Step 1: Matched Context
box1 = patches.FancyBboxPatch((0.15, 1.4), 2.2, 2.2, boxstyle='round,pad=0.12', fc='#E8EAF6', ec=c_shared, lw=1.5)
ax.add_patch(box1)
ax.text(1.25, 3.25, '1. Matched Pair', ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_shared)
ax.text(1.25, 2.7, r'Shared init $\theta(0)$' + '\n' + r'Shared samples $(x_i, y_i)$' + '\n' + r'Shared weak input $x_w$' + '\n' + r'Shared noise & flow', ha='center', va='center', fontsize=8.2, color='#263238', linespacing=1.3)

# Arrow fork
ax.annotate('', xy=(2.7, 3.2), xytext=(2.45, 2.7), arrowprops=dict(arrowstyle='->', color=c_both, lw=1.5))
ax.annotate('', xy=(2.7, 1.8), xytext=(2.45, 2.3), arrowprops=dict(arrowstyle='->', color=c_weak, lw=1.5))

# Step 2: Forked Models
box_both = patches.FancyBboxPatch((2.75, 2.65), 2.1, 1.15, boxstyle='round,pad=0.1', fc='#FFEBEE', ec=c_both, lw=1.5)
ax.add_patch(box_both)
ax.text(3.8, 3.45, 'Both-Feature (B)', ha='center', va='center', fontsize=9, fontweight='bold', color=c_both)
ax.text(3.8, 3.0, r'Strong + Weak cues' + '\n' + r'$\to M_w^{\mathrm{B}}(\tau)$', ha='center', va='center', fontsize=8.2, color='#37474F')

box_weak = patches.FancyBboxPatch((2.75, 1.2), 2.1, 1.15, boxstyle='round,pad=0.1', fc='#E8F5E9', ec=c_weak, lw=1.5)
ax.add_patch(box_weak)
ax.text(3.8, 2.0, 'Weak-Only Control (W)', ha='center', va='center', fontsize=9, fontweight='bold', color=c_weak)
ax.text(3.8, 1.55, r'Strong cue ablated' + '\n' + r'$\to M_w^{\mathrm{W}}(\tau)$', ha='center', va='center', fontsize=8.2, color='#37474F')

# Step 3: Paired Gap & Decomposition
ax.annotate('', xy=(5.2, 2.7), xytext=(4.95, 3.2), arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))
ax.annotate('', xy=(5.2, 2.3), xytext=(4.95, 1.8), arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))

box_gap = patches.FancyBboxPatch((5.3, 1.2), 2.3, 2.6, boxstyle='round,pad=0.12', fc='#F3E5F5', ec=c_diag, lw=1.5)
ax.add_patch(box_gap)
ax.text(6.45, 3.45, '2. Paired Diagnostics', ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_diag)
ax.text(6.45, 2.95, r'$\Delta(\tau) = M_w^{\mathrm{B}}(\tau) - M_w^{\mathrm{W}}(\tau)$' + '\n' + r'(gap in weak response)', ha='center', va='center', fontsize=8.2, fontweight='bold', color='#1A237E')
ax.text(6.45, 2.25, r'$d(\tau) = \dot{\Delta}(\tau)$ (relative drift)' + '\n' + r'$d>0$: transfer; $d<0$: rate supp.', ha='center', va='center', fontsize=7.8, color='#37474F')
ax.text(6.45, 1.55, r'$\Delta(\tau) = P(\tau) - N(\tau)$' + '\n' + r'(pathwise drift balance)', ha='center', va='center', fontsize=7.8, color='#4A148C', fontweight='semibold')

# Step 4: Dual Causal Gate
ax.annotate('', xy=(7.9, 2.5), xytext=(7.7, 2.5), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

box_cert = patches.FancyBboxPatch((8.0, 1.05), 1.85, 2.9, boxstyle='round,pad=0.12', fc='#FFF8E1', ec=c_gate, lw=1.5)
ax.add_patch(box_cert)
ax.text(8.92, 3.65, '3. Causal Decision', ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_gate)

ax.text(8.92, 3.05, r'Gate 1: $\Delta(\tau^*) < 0$' + '\n(Outcome Suppr.)', ha='center', va='center', fontsize=7.6, color='#B71C1C', fontweight='bold')
ax.text(8.92, 2.35, r'Gate 2: $\tau_W^* \in (0, H]$' + '\n(Weak Learnable)', ha='center', va='center', fontsize=7.6, color='#1B5E20', fontweight='bold')

res_pass = patches.Rectangle((8.1, 1.5), 1.65, 0.45, fc='#E8F5E9', ec='#2E7D32', lw=1)
ax.add_patch(res_pass)
ax.text(8.92, 1.72, r'Both Met $\to$ Starvation', ha='center', va='center', fontsize=7.2, fontweight='bold', color='#1B5E20')

res_fail = patches.Rectangle((8.1, 1.15), 1.65, 0.3, fc='#ECEFF1', ec='#78909C', lw=1)
ax.add_patch(res_fail)
ax.text(8.92, 1.3, r'Gate Fails $\to$ Indeterminate', ha='center', va='center', fontsize=6.8, color='#455A64')

# Save outputs
for d in [
    '/Users/prithvirajsangramsinhpatil/Documents/ChatGPT/Gradient-Starvation/overleaf_final_submission/figures',
    '/Users/prithvirajsangramsinhpatil/Documents/ChatGPT/Gradient-Starvation/Gradient-Starvation-Research/paper/figures',
    '/Users/prithvirajsangramsinhpatil/Documents/ChatGPT/Gradient-Starvation/overleaf_paper/figures'
]:
    os.makedirs(d, exist_ok=True)
    outpath = os.path.join(d, 'fig1_causal_schematic.pdf')
    fig.savefig(outpath, bbox_inches='tight')
    print('Saved', outpath)

print('Done!')
