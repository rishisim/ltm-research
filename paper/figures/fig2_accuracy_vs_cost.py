#!/usr/bin/env python3
"""Figure 2: Accuracy vs Step Cost scatter — held-out splits (unseen + test)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data — held-out pair: ALFWorld unseen + WebShop test
frameworks = ['ReAct', 'Reflexion', 'CR', 'TR', 'CR+TR']

alf_acc   = [61.94, 83.58, 76.12, 70.15, 81.34]
alf_steps = [22.38, 74.90, 22.55, 24.93, 23.35]

web_acc   = [50.50, 60.00, 31.50, 34.00, 23.00]
web_steps = [27.97, 107.38, 29.62, 25.92, 31.36]

# Style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, ax = plt.subplots(figsize=(6, 5))

# ALFWorld points (circles, blue)
ax.scatter(alf_steps, alf_acc, s=90, marker='o', color='#2b8cbe',
           edgecolors='black', linewidth=0.5, zorder=5, label='ALFWorld (unseen)')

# WebShop points (squares, red)
ax.scatter(web_steps, web_acc, s=90, marker='s', color='#e34a33',
           edgecolors='black', linewidth=0.5, zorder=5, label='WebShop (test)')

# Label each point
offsets_alf = {
    'ReAct':     (-8, -12),
    'Reflexion': (-10, 6),
    'CR':        (6, -10),
    'TR':        (6, 4),
    'CR+TR':     (-14, -12),
}
offsets_web = {
    'ReAct':     (-8, 6),
    'Reflexion': (-10, -12),
    'CR':        (6, 4),
    'TR':        (-10, -12),
    'CR+TR':     (6, -4),
}

for i, fw in enumerate(frameworks):
    ox, oy = offsets_alf[fw]
    ax.annotate(fw, (alf_steps[i], alf_acc[i]), textcoords='offset points',
                xytext=(ox, oy), fontsize=9, color='#2b8cbe', fontweight='bold')
    ox, oy = offsets_web[fw]
    ax.annotate(fw, (web_steps[i], web_acc[i]), textcoords='offset points',
                xytext=(ox, oy), fontsize=9, color='#e34a33', fontweight='bold')

# Annotation arrow: CR+TR near-Reflexion accuracy at 1/3 cost (ALFWorld)
ax.annotate(
    'Near-Reflexion accuracy\nat ~1/3 the cost',
    xy=(23.35, 81.34),
    xytext=(48, 72),
    fontsize=9, fontstyle='italic',
    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
    bbox=dict(boxstyle='round,pad=0.3', fc='#f0f0f0', ec='gray', alpha=0.8),
)

ax.set_xlabel('Avg Steps / Task', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(15, 95)
ax.set_xlim(15, 115)
ax.legend(fontsize=10, loc='lower right', frameon=False)

plt.tight_layout()
plt.savefig('/Users/rishisim/Documents/research/ltm-research/paper/figures/accuracy_vs_cost.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Saved accuracy_vs_cost.pdf")
