import numpy as np
import time
np.seterr(all='raise')
import matplotlib.image as mpimg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class Plot():
    def __init__(self, wheel):
        self.wheel = wheel
        self.fig = plt.figure(figsize=(14, 18))
        figtext1 = "Edge Size: {0}\n\
Opacity: {1}\n\
Lookahead: {4}\n\
Number of points: {2}\n\
Padding: {3}\n".format(self.wheel.EDGE_SIZE,
                       self.wheel.LINE_OPACITY,
                       self.wheel.NUM_POINTS,
                       self.wheel.PADDING,
                       self.wheel.LOOKAHEAD)
        plt.figtext(.15, 0, figtext1, color='black',
                    weight='roman', size=16)

        figtext2 = "Random Interval: {0}\n\
Cache Refresh Interval: {2}\n\
stopping_interval: {1}".format(self.wheel.RANDOM_INTERVAL,
                               self.wheel.STOPPING_LOOKBACK,
                               self.wheel.CACHE_REFRESH)
        plt.figtext(.5, 0.03, figtext2, color='black',
                    weight='roman', size=16)

        self.ax1 = plt.subplot2grid((6, 4), (0, 0))
        self.ax1.set_title("Target Image")

        self.ax8 = plt.subplot2grid((6, 4), (1, 0))
        self.ax8.set_title("Weights")

        self.ax2 = plt.subplot2grid((6, 4), (0, 1))
        self.ax2.set_title("Non-rasterized threads")
        self.ax2.set_xlim([0, self.wheel.EDGE_SIZE])
        self.ax2.set_ylim([self.wheel.EDGE_SIZE, 0])
        self.ax2.set_aspect('equal')

        self.ax3 = plt.subplot2grid((6, 4), (1, 1))
        self.ax3.set_title("Rasterized threads")

        self.ax4 = plt.subplot2grid((6, 4), (0, 2))
        self.ax4.set_title("L1 Error")

        self.ax9 = plt.subplot2grid((6, 4), (1, 2))
        self.ax9.set_title("Weighted L1 Error")

        self.ax10 = plt.subplot2grid((6, 4), (0, 3))
        self.ax10.set_title("L2 Error")

        self.ax11 = plt.subplot2grid((6, 4), (1, 3))
        self.ax11.set_title("Weighted L2 Error")

        self.ax5 = plt.subplot2grid((6, 4), (3, 0), colspan=4)
        self.ax5.set_title("Error improvement")

        self.ax12 = plt.subplot2grid((6, 4), (2, 0), colspan=2)
        self.ax12.set_title("L1 Error")

        self.ax13 = plt.subplot2grid((6, 4), (2, 2), colspan=2)
        self.ax13.set_title("L2 Error")

        self.ax6 = plt.subplot2grid((6, 4), (4, 0), colspan=4)
        self.ax6.set_title("Pin number")

        self.ax7 = plt.subplot2grid((6, 4), (5, 0), colspan=4)
        self.ax7.set_title("Line length")
        self.last_line = 0
        self.ax1.imshow(self.wheel.img, cmap="gray", vmin=0, vmax=255)
        self.ax8.imshow(self.wheel.weights, cmap="gray", vmin=0, vmax=255)

    def show(self, lengths):
        for i in xrange(self.last_line, len(self.wheel.points_log) - 1):
            line_x = [self.wheel.points[self.wheel.points_log[i]][0],
                      self.wheel.points[self.wheel.points_log[i + 1]][0]]
            line_y = [self.wheel.points[self.wheel.points_log[i]][1],
                      self.wheel.points[self.wheel.points_log[i + 1]][1]]
            self.ax2.plot(line_y, line_x,
                          alpha=self.wheel.LINE_OPACITY, color="k")
        # self.ax2.plot(*zip(*points), 'b.')
        self.ax3.imshow(self.wheel.raster, cmap="gray", vmin=0, vmax=255)
        weighted_diff = np.multiply(self.wheel.weights / 128, self.wheel.diff)
        self.ax4.imshow(self.wheel.diff, cmap="seismic")
        self.ax9.imshow(weighted_diff, cmap="seismic")

        self.ax10.imshow(np.square(self.wheel.diff), cmap="seismic")
        self.ax11.imshow(np.multiply(np.square(weighted_diff),
                                     np.sign(self.wheel.diff)), cmap="seismic")
        self.fig.canvas.draw()
        # self.ax1.plot(range(self.last_line,len(self.wheel.points_log)-1),self.wheel.loss_delta[self.last_line:])
        self.ax12.plot(range(self.last_line, len(
            self.wheel.points_log) - 1), self.wheel.l1_errors[self.last_line:])
        self.ax13.plot(range(self.last_line, len(
            self.wheel.points_log) - 1), self.wheel.l2_errors[self.last_line:])
        self.ax5.plot(range(self.last_line, len(
            self.wheel.points_log) - 1), self.wheel.loss_delta[self.last_line:])
        self.ax6.plot(range(self.last_line, len(
            self.wheel.points_log) - 1), self.wheel.points_log[self.last_line:-1])
        self.ax7.plot(range(self.last_line, len(
            self.wheel.points_log) - 1), lengths[self.last_line:])
        self.last_line = len(self.wheel.points_log)
