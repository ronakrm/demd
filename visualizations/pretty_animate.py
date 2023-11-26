# Figure out how to make pretty gifs with matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

myBlue = '#005baa'
myRed = '#ff2a2a'
myPurple = '#25a000'
myGreen = '#ed00d4'
myColors = [myBlue, myRed, myPurple, myGreen]

cur_theta = 0

fig = plt.figure(frameon=False)
ax = fig.add_axes([0, 0, 1, 1])

# clear everything except plot area
ax.axis('off')
for item in [fig, ax]:
    item.patch.set_visible(False)

def plotter(param):

    x = np.arange(0, 5*np.pi, 0.1)
    y = np.sin(x+param)

    plt.plot(x, y)

def animate(i=0):
    cur_theta = i*0.1
    plotter(param=cur_theta)

#    g.savefig(f'step_{i}_cost_{emdi:.2f}.png')

anim = animation.FuncAnimation(fig,
                                animate,
                                init_func=animate,
                                interval=1,
                                frames=30,
                                repeat=True
                                )
anim.save("pretty_test.gif",
            codec="png",
            bitrate=-1
            #savefig_kwargs={"transparent": True, "facecolor": "none"},
            )
