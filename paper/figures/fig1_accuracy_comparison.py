#!/usr/bin/env python3
"""Figure 1: Accuracy Comparison — two-panel bar chart with proper split pairing."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data
frameworks = ['ReAct', '+Reflexion', '+CR', '+TR', '+CR+TR', 'hard-neg']

# Development pair: ALFWorld seen + WebShop dev
alf_seen  = [61.43, 82.14, 77.14, 63.57, 77.86, 55.71]
web_dev   = [56.00, 67.00, 27.00, 34.50, 25.50, 43.50]

# Held-out pair: ALFWorld unseen + WebShop test
alf_unseen = [61.94, 83.58, 76.12, 70.15, 81.34, 44.78]
web_test   = [50.50, 60.00, 31.50, 34.00, 23.00, 35.50]

x = np.arange(len(frameworks))
width = 0.35

# Style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

# --- Left panel: Development splits ---
b1 = ax1.bar(x - width/2, alf_seen, width, label='ALFWorld (seen)',
             color='#2b8cbe', edgecolor='white', linewidth=0.5)
b2 = ax1.bar(x + width/2, web_dev, width, label='WebShop (dev)',
             color='#e34a33', edgecolor='white', linewidth=0.5)

for bar in b1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 1.0, f'{h:.1f}',
             ha='center', va='bottom', fontsize=7.5)
for bar in b2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 1.0, f'{h:.1f}',
             ha='center', va='bottom', fontsize=7.5)

ax1.axhline(y=61.43, color='#2b8cbe', linestyle='--', linewidth=0.8, alpha=0.4)
ax1.axhline(y=56.00, color='#e34a33', linestyle='--', linewidth=0.8, alpha=0.4)

ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(frameworks, fontsize=9, rotation=15, ha='right')
ax1.set_ylim(0, 100)
ax1.set_title('Development splits', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right', frameon=False)

# --- Right panel: Held-out splits ---
b3 = ax2.bar(x - width/2, alf_unseen, width, label='ALFWorld (unseen)',
             color='#2b8cbe', edgecolor='white', linewidth=0.5)
b4 = ax2.bar(x + width/2, web_test, width, label='WebShop (test)',
             color='#e34a33', edgecolor='white', linewidth=0.5)

for bar in b3:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 1.0, f'{h:.1f}',
             ha='center', va='bottom', fontsize=7.5)
for bar in b4:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 1.0, f'{h:.1f}',
             ha='center', va='bottom', fontsize=7.5)

ax2.axhline(y=61.94, color='#2b8cbe', linestyle='--', linewidth=0.8, alpha=0.4)
ax2.axhline(y=50.50, color='#e34a33', linestyle='--', linewidth=0.8, alpha=0.4)

ax2.set_xticks(x)
ax2.set_xticklabels(frameworks, fontsize=9, rotation=15, ha='right')
ax2.set_ylim(0, 100)
ax2.set_title('Held-out splits', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right', frameon=False)

plt.tight_layout()
plt.savefig('/Users/rishisim/Documents/research/ltm-research/paper/figures/accuracy_comparison.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Saved accuracy_comparison.pdf")
