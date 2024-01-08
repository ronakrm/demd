from turtle import right
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

myBlue = '#005baa'
myRed = '#ff2a2a'
myPurple = '#25a000'
myGreen = '#ed00d4'
myColors = [myBlue, myRed, myPurple, myGreen]

def clearAxisDecorators(ax):
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    ax.tick_params(bottom=False,left=False)
    ax.set(xlabel=None)
    ax.set(ylabel=None)

def jointGridplot(g, a, b, joint, plim=0.5, show_vals=False):

    n = len(a)
    bin_boundaries = range(n+1)

    g.ax_marg_x.clear()
    g.ax_marg_y.clear()
    g.ax_joint.clear()


    g.ax_marg_x.set(ylim=(0,plim))
    axa = sns.histplot(x=range(n), 
                       weights=a,
                       stat='count',
                       bins=bin_boundaries,
                       ax=g.ax_marg_x,
                       color=myBlue,
                       alpha=1
                    )

    if show_vals:
        for p, val in zip(axa.patches, a):
            axa.text(p.get_x() + p.get_width()/2.,
                    -0.25,#plim,
                    '{}'.format(val),
                    fontsize=12,
                    ha='center',
                    va='bottom')

    g.ax_marg_y.set(xlim=(0,plim))
    axb = sns.histplot(y=range(n),
                       weights=b, 
                       stat='count',
                       bins=bin_boundaries,
                       ax=g.ax_marg_y,
                       color=myRed,
                       alpha=1
                     )

    if show_vals:
        for p, val in zip(axb.patches, b):
            axb.text(-0.25,#plim,
                     p.get_y() + p.get_height()/2.,
                    '{}'.format(val),
                    fontsize=12,
                    ha='center',
                    va='bottom')


    sns.heatmap(joint.transpose(),
                vmin=0, vmax=plim,
                annot=show_vals,
                cbar=False,
                ax=g.ax_joint)

    clearAxisDecorators(g.ax_marg_x)
    clearAxisDecorators(g.ax_marg_y)
    clearAxisDecorators(g.ax_joint)

    g.ax_marg_x.set_frame_on(False)
    g.ax_marg_y.set_frame_on(False)
    g.ax_joint.set_frame_on(False)

def giffer(logs, outName, plim=0.5, show_vals=False):

    g = sns.JointGrid(space=0.75)

    def animate(i=0):
        ai = logs[i]['a']
        bi = logs[i]['b']
        dpmati = logs[i]['dpmat']
        emdi = logs[i]['emd']

        jointGridplot(g, ai, bi, dpmati, plim=plim, show_vals=show_vals)
    #    g.savefig(f'step_{i}_cost_{emdi:.2f}.png')
    
    anim = animation.FuncAnimation(g.figure,
                                   animate,
                                   init_func=animate,
                                   interval=5000/len(logs),
                                   frames=len(logs),
                                   repeat=False
                                   )
    anim.save(outName,
              codec="png",
              bitrate=-1
              #savefig_kwargs={"transparent": True, "facecolor": "none"},
              )
def univariateGiffer(logs, outName, plim=None, show_vals=False):

    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.linewidth"]  = 0.0

    fig = plt.figure(figsize=(9,3), dpi=300, frameon=False)
    plt.tight_layout(pad=0.0)
    ax = fig.add_axes([0, 0, 1, 1])

    # clear everything except plot area
    ax.axis('off')
    for item in [fig, ax]:
        item.patch.set_visible(False)
        item.patch.set_linewidth(0.0)
    #plt.subplots_adjust(wspace=0, hspace=0)
    ax.set_axis_off()
    ax.set_frame_on(False)
    ax.minorticks_off()
    plt.tick_params(
        which='both',      # both major and minor ticks are affected
        right=False,
        left=False,      # ticks along the bottom edge are off
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    def jointSingleAxisPlot(A: np.array, show_vals=False):

        n, d = A.shape
        bin_boundaries = range(d+1)

        #colors = sns.set_palette("tab10", n=n)

        for i in range(n):
            sns.histplot(x=range(d),
                                weights=A[i,:], 
                                stat='count',
                                bins=bin_boundaries,
                                color=myColors[i],
                                alpha=0.5 + 0.5*((n-i)/n),
                                ax=ax,
                                legend=False
                                )
        ax.set_xlim(0, d)

    def animate(i=0):
        ax.clear()
        ax.patch.set_visible(False)
        if plim is not None:
            ax.set_ylim([0,plim])
        #plt.subplots_adjust(wspace=0, hspace=0)
        Ai = logs[i]['A']

        jointSingleAxisPlot(Ai, show_vals=show_vals)
        #import pdb; pdb.set_trace()
        #fig.savefig(f'test_single.png')
    
    anim = animation.FuncAnimation(fig,
                                   animate,
                                   init_func=animate,
                                   interval=5000/len(logs),
                                   frames=len(logs),
                                   repeat=False
                                   )
    anim.save(outName,
              codec="png",
              bitrate=-1,
              savefig_kwargs={"transparent": True,
                              #"bbox_inches": 'tight',
                              },
              )