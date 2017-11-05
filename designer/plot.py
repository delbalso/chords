import numpy as np
np.seterr(all='raise')
import matplotlib.image as mpimg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class Plot():
    def __init__(self, wheel, img, weights, points):
        self.wheel = wheel
        self.points = points
        self.fig = plt.figure(figsize=(12, 16))
        figtext = "edge_size: {0}\nopacity: {1}\nnum_points: {2}\npadding: {3}\nrandom_interval: {4}\nstopping_interval: {5}".format(self.wheel.EDGE_SIZE, self.wheel.LINE_OPACITY, self.wheel.NUM_POINTS, self.wheel.PADDING, self.wheel.RANDOM_INTERVAL, self.wheel.STOPPING_LOOKBACK)
        plt.figtext(0, 0.9, figtext, color='black', weight='roman',
            size='small')

        self.ax1 = plt.subplot2grid((5, 3), (0, 0))
        self.ax1.set_title("Target Image")

        self.ax8 = plt.subplot2grid((5, 3), (1, 0))
        self.ax8.set_title("Weights")

        self.ax2 = plt.subplot2grid((5, 3), (0, 1))
        self.ax2.set_title("Non-rasterized threads")
        self.ax2.set_xlim([0, self.wheel.EDGE_SIZE])
        self.ax2.set_ylim([self.wheel.EDGE_SIZE, 0])
        self.ax2.set_aspect('equal')

        self.ax3 = plt.subplot2grid((5, 3), (1, 1))
        self.ax3.set_title("Rasterized threads")

        self.ax4 = plt.subplot2grid((5, 3), (0, 2))
        self.ax4.set_title("Error")

        self.ax9 = plt.subplot2grid((5, 3), (1, 2))
        self.ax9.set_title("Weighted Error")

        self.ax5 = plt.subplot2grid((5, 3), (2, 0), colspan=3)
        self.ax5.set_title("Error improvement")

        self.ax6 = plt.subplot2grid((5, 3), (3, 0), colspan=3)
        self.ax6.set_title("Pin number")

        self.ax7 = plt.subplot2grid((5, 3), (4, 0), colspan=3)
        self.ax7.set_title("Line length")
        self.last_line = 0
        self.ax1.imshow(img, cmap="gray", vmin=0, vmax=255)
        self.ax8.imshow(weights, cmap="gray", vmin=0, vmax=255)

        #self.fig.subplots(figsize=(20, 10))
        #plt.tight_layout()

    def show(self, weights, raster, diff, points_log,errors, pins, lengths):
        self.ax9.imshow(np.multiply(weights/128,diff), cmap="gray", vmin=0, vmax=255)
        #lc = LineCollection([points_log[-1]], linewidths=2, alpha=0.01)
        #self.ax2.add_collection(lc)
        for i in xrange(self.last_line, len(points_log)-1):
            line_x = [self.points[points_log[i]][0], self.points[points_log[i+1]][0]]
            line_y = [self.points[points_log[i]][1], self.points[points_log[i+1]][1]]
            self.ax2.plot(line_y, line_x, alpha=self.wheel.LINE_OPACITY, color="k")
        #self.ax2.plot(*zip(*points), 'b.')
        self.ax3.imshow(raster, cmap="gray", vmin=0, vmax=255)
        self.ax4.imshow(diff, cmap="gray", vmin=0, vmax=255)
        self.fig.canvas.draw()
        self.ax5.plot(range(self.last_line,len(points_log)-1),errors[self.last_line:])
        self.ax6.plot(range(self.last_line,len(points_log)-1),pins[self.last_line:-1])
        self.ax7.plot(range(self.last_line,len(points_log)-1),lengths[self.last_line:])
        self.last_line = len(points_log)
