#!/usr/bin/env python3
"""Figure 3: Delta from ReAct baseline — all 4 splits, grouped by development vs held-out."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data (pp change from ReAct baseline)
frameworks = ['+CR', '+TR', '+CR+TR', '+Reflexion', 'hard-neg']

# Development pair
alf_seen   = [15.71, 2.14, 16.43, 20.71, -5.71]
web_dev    = [-29.00, -21.50, -30.50, 11.00, -12.50]

# Held-out pair
alf_unseen = [14.18, 8.21, 19.40, 21.64, -17.16]
web_test   = [-19.00, -16.50, -27.50, 9.50, -15.00]

y = np.arange(len(frameworks))
height = 0.19

# Style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, ax = plt.subplots(figsize=(8, 5.5))

# Color helper: green-ish for positive, red-ish for negative
def bar_colors(vals, base_pos, base_neg):
    return [base_pos if v >= 0 else base_neg for v in vals]

# 4 bars per framework, ordered: alf_seen, web_dev, alf_unseen, web_test (development pair then held-out pair)
bars1 = ax.barh(y + 1.5*height, alf_seen, height, label='ALFWorld (seen)',
                color=bar_colors(alf_seen, '#66c2a5', '#fc8d62'),
                edgecolor='white', linewidth=0.5)
bars2 = ax.barh(y + 0.5*height, web_dev, height, label='WebShop (dev)',
                color=bar_colors(web_dev, '#a6d854', '#e31a1c'),
                edgecolor='white', linewidth=0.5)
bars3 = ax.barh(y - 0.5*height, alf_unseen, height, label='ALFWorld (unseen)',
                color=bar_colors(alf_unseen, '#2b8cbe', '#e34a33'),
                edgecolor='white', linewidth=0.5)
bars4 = ax.barh(y - 1.5*height, web_test, height, label='WebShop (test)',
                color=bar_colors(web_test, '#78c679', '#cb181d'),
                edgecolor='white', linewidth=0.5)

# Value labels
def add_labels(bars, vals):
    for bar, v in zip(bars, vals):
        xpos = v + (0.8 if v >= 0 else -0.8)
        ha = 'left' if v >= 0 else 'right'
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f'{v:+.1f}', va='center', ha=ha, fontsize=7.5)

add_labels(bars1, alf_seen)
add_labels(bars2, web_dev)
add_labels(bars3, alf_unseen)
add_labels(bars4, web_test)

# Zero line
ax.axvline(x=0, color='black', linewidth=0.8)

ax.set_yticks(y)
ax.set_yticklabels(frameworks, fontsize=11)
ax.set_xlabel('Gain from ReAct baseline (pp)', fontsize=12)
ax.legend(fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, 1.08),
          frameon=False, ncol=4)

plt.tight_layout()
plt.savefig('/Users/rishisim/Documents/research/ltm-research/paper/figures/gain_from_baseline.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Saved gain_from_baseline.pdf")
