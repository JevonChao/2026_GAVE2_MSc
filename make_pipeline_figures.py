"""
生成三张技术路线图 (Task 1 / 2 / 3),matplotlib 绘制,风格与结果图统一。
产出: fig_pipeline_task1.png / fig_pipeline_task2.png / fig_pipeline_task3.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

BLUE = '#4C72B0'
ORANGE = '#DD8452'
GREEN = '#55A868'
RED = '#C44E52'
GREY = '#8C8C8C'


def box(ax, x, y, w, h, text, fc='white', ec='black', fs=10, tc='black'):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle='round,pad=0.02,rounding_size=0.06',
                       linewidth=1.2, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=tc, wrap=True)


def arrow(ax, x1, y1, x2, y2, color='black'):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=14,
                        linewidth=1.2, color=color, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def new_ax(w=9, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis('off')
    return fig, ax


# ---------- Task 1 ----------
fig, ax = new_ax()
box(ax, 1.2, 1.7, 1.8, 1.0, 'CFP\nimage\n(3-ch)', fc='#EAF0F8', ec=BLUE)
box(ax, 3.7, 1.7, 1.9, 1.0, 'Base U-Net\n(first_u)', fc='white', ec=BLUE)
box(ax, 6.2, 1.7, 2.0, 1.0, 'Recursive\nrefinement\n(×5)', fc='white', ec=BLUE)
box(ax, 8.7, 1.7, 1.8, 1.0, 'A / V / vessel\nprediction', fc='#EAF0F8', ec=BLUE)
arrow(ax, 2.1, 1.7, 2.75, 1.7)
arrow(ax, 4.65, 1.7, 5.2, 1.7)
arrow(ax, 7.2, 1.7, 7.8, 1.7)
# refinement self-loop hint
ax.annotate('', xy=(5.7, 2.35), xytext=(6.7, 2.35),
            arrowprops=dict(arrowstyle='<-', color=GREY, lw=1,
                            connectionstyle='arc3,rad=-0.5'))
ax.text(6.2, 2.7, 'shared weights', ha='center', fontsize=8, color=GREY, style='italic')
fig.tight_layout()
fig.savefig('fig_pipeline_task1.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('saved fig_pipeline_task1.png')


# ---------- Task 2 ----------
fig, ax = new_ax(w=9.2, h=4.0)
ax.set_ylim(0, 4.4)
# three inputs
box(ax, 1.2, 3.4, 1.7, 0.8, 'CFP (3-ch)', fc='#EAF0F8', ec=BLUE, fs=9)
box(ax, 1.2, 2.3, 1.7, 0.8, 'FFA_A (1-ch)', fc='#FBEEE4', ec=ORANGE, fs=9)
box(ax, 1.2, 1.2, 1.7, 0.8, 'FFA_AV (1-ch)', fc='#FBEEE4', ec=ORANGE, fs=9)
# encoders
box(ax, 3.5, 3.4, 1.8, 0.8, 'RGB encoder', fc='white', ec=BLUE, fs=9)
box(ax, 3.5, 1.75, 1.8, 1.4, 'Shared FFA\nencoder', fc='white', ec=ORANGE, fs=9)
arrow(ax, 2.05, 3.4, 2.6, 3.4)
arrow(ax, 2.05, 2.3, 2.6, 2.0)
arrow(ax, 2.05, 1.2, 2.6, 1.5)
# fusion
box(ax, 6.0, 2.55, 2.0, 1.4, 'Multi-scale\nfusion\n(4 levels)', fc='#FDF3E1', ec=RED, fs=9)
arrow(ax, 4.4, 3.4, 5.0, 2.85)
arrow(ax, 4.4, 1.75, 5.0, 2.25)
# refinement + output
box(ax, 8.5, 3.4, 1.6, 0.8, 'Refinement\n(×5)', fc='white', ec=BLUE, fs=9)
box(ax, 8.5, 1.8, 1.6, 0.8, 'A / V / vessel', fc='#EAF0F8', ec=BLUE, fs=9)
arrow(ax, 7.0, 2.7, 7.7, 3.2)
arrow(ax, 8.5, 3.0, 8.5, 2.2)
ax.text(6.0, 1.55, 'add / weighted / attention', ha='center', fontsize=8,
        color=RED, style='italic')
fig.tight_layout()
fig.savefig('fig_pipeline_task2.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('saved fig_pipeline_task2.png')


# ---------- Task 3 ----------
fig, ax = new_ax(w=9.4, h=3.4)
box(ax, 1.3, 2.4, 1.9, 0.9, 'A/V segmentation\n(from Task 1/2)', fc='#EAF0F8', ec=BLUE, fs=9)
box(ax, 1.3, 1.0, 1.9, 0.9, 'Optic disc\nlocalisation', fc='#EFEFEF', ec=GREY, fs=9)
box(ax, 4.1, 1.7, 2.0, 1.2, 'Annular zones\n(0.5 / 1.0 / 2.0 DD)', fc='white', ec=GREEN, fs=9)
# inputs -> annular zones (annular left edge = 3.1)
arrow(ax, 2.25, 2.4, 3.1, 1.9)
arrow(ax, 2.25, 1.0, 3.1, 1.5)
# three biomarker boxes (left edge = 5.7)
box(ax, 6.7, 2.55, 2.0, 0.9, 'Knudtson formula\n→ CRAE, CRVE, AVR', fc='#EAF3EC', ec=GREEN, fs=8.5)
box(ax, 6.7, 1.5, 2.0, 0.7, 'Density (zone C)', fc='#EAF3EC', ec=GREEN, fs=9)
box(ax, 6.7, 0.55, 2.0, 0.7, 'Fractal dimension', fc='#EAF3EC', ec=GREEN, fs=9)
# annular (right edge = 5.1) -> three boxes (left edge = 5.7)
arrow(ax, 5.1, 1.9, 5.7, 2.55)
arrow(ax, 5.1, 1.7, 5.7, 1.5)
arrow(ax, 5.1, 1.5, 5.7, 0.55)
# final box (left edge = 8.4)
box(ax, 9.1, 1.5, 1.4, 0.9, '7 vascular\nbiomarkers', fc='#EAF3EC', ec=GREEN, fs=9)
# three boxes (right edge = 7.7) -> final (left edge = 8.4)
arrow(ax, 7.7, 2.55, 8.4, 1.75)
arrow(ax, 7.7, 1.5, 8.4, 1.5)
arrow(ax, 7.7, 0.55, 8.4, 1.25)
fig.tight_layout()
fig.savefig('fig_pipeline_task3.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('saved fig_pipeline_task3.png')

print('\nAll pipeline diagrams generated.')
