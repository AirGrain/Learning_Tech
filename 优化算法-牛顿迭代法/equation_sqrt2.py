import numpy as np
import matplotlib.pyplot as plt

import mpl_toolkits.axisartist as axisxy


x = np.linspace(-8,8,1000)
y = np.square(x)-2
# x=5时候的切线
x2 = np.linspace(2,6,1000)
y2 = 10*x2-27

fig = plt.figure(figsize=(4,4))
ax = axisxy.Subplot(fig,111)
fig.add_axes(ax)
ax.axis[:].set_visible(False)
ax.axis["x"] = ax.new_floating_axis(0,0)
ax.axis["x"].set_axisline_style("->",size = 1.0)

ax.axis["y"] = ax.new_floating_axis(1,0)
ax.axis["y"].set_axisline_style("-|>", size = 1.0)

ax.axis["x"].set_axis_direction("top")
ax.axis["y"].set_axis_direction("right")

plt.plot(x,y,'b')
plt.plot(x2,y2,'r')  
plt.show()
