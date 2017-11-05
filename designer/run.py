import numpy as np
np.seterr(all='raise')
from bresenham import bresenham
import itertools
from scipy import ndimage, stats
from scipy.misc import imresize
from random import randint
import pickle
import time
import os

import plot

def get_image(f):
    img = ndimage.imread(f, flatten=True)
    return img

def normalize_image(img, edge_size):
    return imresize(img.astype('uint8'),(edge_size, edge_size)).astype('float32')

class WheelOptimizer:
    def __init__(self, wheel):
        self.wheel = wheel
        self.line_scores = {}

    # update_line_scores updates line_scores for a given line index=(p1,p2)
    def update_line_scores(self,index):
        a,b = index
        if a>b: a,b=b,a
        index = a,b
        pixels = self.wheel.ref_lines[index]
        score = 0
        l1=0
        l2=0

        #self.line_scores[index] = 0.5
        #return
        # calculate score
        for pixel in pixels:
            l1+=0 # diff[pixel] * 10
            l2+=self.wheel.diff[pixel]**2 * np.sign(self.wheel.diff[pixel])
            score += self.wheel.weights[pixel] * (l1 + l2)
        self.line_scores[index] = float(score)/len(pixels)

    #refresh_line_score runs update_line_scores on all lines effectively wiping the cache
    def refresh_line_scores(self):
        cache_start_time = time.time()
        for index in self.wheel.ref_lines.keys():
            self.update_line_scores(index)
        print "Refreshing cache took {0} seconds".format(time.time() - cache_start_time)

    # get_next_point figures out what the best point is to use next given a certain state
    # state of the wheel/optimizer
    def get_next_point(self, prev, lookahead=0):
        best_score = None
        starting_point =  prev[-1]

        # Determine if we're going to use a random point or not
        if self.wheel.RANDOM_INTERVAL!=None and randint(1, self.wheel.RANDOM_INTERVAL)==1:
            using_random_point = True
            random_choice = randint(0, self.wheel.NUM_POINTS-1)
        else:
            using_random_point = False

        keys = []
        for i in xrange(self.wheel.NUM_POINTS):
            # Reject lines that are too short
            if (prev[-1]-i)%self.wheel.NUM_POINTS < self.wheel.PADDING or (prev[-1]-i)%self.wheel.NUM_POINTS > self.wheel.NUM_POINTS - self.wheel.PADDING: continue
            # Reject lines going directly back to where they came from
            if (len(prev)>1 and prev[-2]==i): continue

            # Get the score of the proposed_line
            a,b = prev[-1],i
            if a>b: a,b=b,a
            score = self.line_scores[a,b]

            if lookahead>0: #TODO fix how this is done to normalize by length after adding together
                score += self.get_next_point([prev[-1],i], lookahead=lookahead-1)[1]

            if ((using_random_point and i==random_choice) or
                best_score is None or
                score>best_score):
                best_i=i
                best_score = score

            if using_random_point and i==random_choice:
                break
        return best_i, best_score

class Wheel:
    def __init__(self):
        # Config parameters
        self.EDGE_SIZE = 500
        self.NUM_POINTS = 200
        self.LOOKAHEAD = 1
        self.STOPPING_LOOKBACK = 10
        self.LINE_OPACITY = .03
        self.PADDING = 10
        self.RANDOM_INTERVAL = None
        self.CACHE_REFRESH = 20
        self.PLOT_INTERVAL = 50

        self.original_img = get_image('/Users/delbalso/Downloads/dave.jpg')
        self.original_weights = get_image('/Users/delbalso/Downloads/dave-mask.jpg')

        self.points = self.config_circle()

        self.ref_lines = {}
        for a,b in list(itertools.combinations(xrange(self.NUM_POINTS),2)):
            if a>b: a,b=b,a
            self.ref_lines[a,b] = list(bresenham(self.points[a][0], self.points[a][1], self.points[b][0], self.points[b][1]))


        # Set up main images
        self.img = normalize_image((self.original_img - self.original_img.min())/self.original_img.max()*255, self.EDGE_SIZE)
        self.weights = normalize_image(self.original_weights, self.EDGE_SIZE)
        self.p = plot.Plot(self, self.img, self.weights, self.points)

        # Set up derivative images
        # Raster is the image we're drawing to simulate thread
        self.raster = np.zeros((self.EDGE_SIZE, self.EDGE_SIZE))+255
        assert self.raster.shape == self.img.shape
        self.diff = np.subtract(self.raster, self.img)

        # start with a random point
        self.points_log = [randint(0, self.NUM_POINTS-1)]



        self.process_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        self.wheelOptimizer = WheelOptimizer(self)

    def config_circle(self):
        # Circle Points
        theta = np.linspace(0, 2*np.pi, self.NUM_POINTS)
        a, b = self.EDGE_SIZE * (np.cos(theta)+1)/2-1, self.EDGE_SIZE * (np.sin(theta)+1)/2-1
        return zip([int(i) for i in a],[int(i) for i in b])


    # calculate runs the logic to figure out the right weaving
    def calculate(self):
        errors = []
        times = []
        lengths = []
        i = 0
        while (len(errors)<self.STOPPING_LOOKBACK or
               sum(errors[-self.STOPPING_LOOKBACK:]) > self.STOPPING_LOOKBACK*0.125): # TODO cleanup this

            if i%5 == 0: print "Starting step {0}".format(i)
            if i%self.CACHE_REFRESH == 0:
                self.wheelOptimizer.refresh_line_scores()

            start_time = time.time() # Start timer

            # Get next Point
            next_point, err = self.wheelOptimizer.get_next_point(self.points_log[-1:],
                                                                 lookahead=self.LOOKAHEAD)
            self.points_log.append(next_point)
            errors.append(err)

            # Get Pixels of new line
            new_line_pixels = self.get_line_pixels(self.points_log[-2],self.points_log[-1])
            lengths.append(len(new_line_pixels))

            # Draw the line
            self.draw_line(new_line_pixels)
            self.update_diff(new_line_pixels)
            self.wheelOptimizer.update_line_scores((self.points_log[-2],self.points_log[-1]))

            times.append(time.time() - start_time) # End timer

            if i%self.PLOT_INTERVAL==1:
                self.p.show(self.weights,self.raster,self.diff,self.points_log, errors, self.points_log, lengths)
                self.save_snapshot(i)
                print(stats.describe(np.array(times)))

            i = i+1
        print "Finished! We drew {0} lines".format(i)

    def update_diff(self, pixels): # optimization?
        for pixel in pixels:
            self.diff[pixel] = self.raster[pixel] - self.img[pixel]

    def save_snapshot(self, steps):
        save_dir = os.path.join("results",self.process_start_time, str(steps))
        try:
            os.makedirs(save_dir)
        except:
            pass

        pickle.dump({"point_log": self.points_log,
                     "image": self.original_img,
                     "weights": self.original_weights,
                     "figure": self.p},
                    open(os.path.join(save_dir,"{0}.p".format(steps)), "wb" ))
        self.p.fig.savefig(os.path.join(save_dir,"{0}.png".format(steps)))

    # Darken a set (line) of pixels on the raster image
    def draw_line(self,pixels):
        for pixel in pixels:
            self.raster[pixel[0],pixel[1]] = max(self.raster[pixel[0],pixel[1]] * (float(1)-self.LINE_OPACITY),0)

    # For a given line between p1 and p2, what pixels should be drawn on
    def get_line_pixels(self, p1, p2, thickness = 5):
        line_points = []
        if p1>p2: p1,p2=p2,p1
        line_px = self.ref_lines[p1,p2]
        p1 = self.points[p1]
        p2 = self.points[p2]
        for px in line_px:
            if abs(p1[0]-p2[0])>abs(p1[1]-p2[1]):
                for j in xrange(thickness):
                    if px[1]+j>=self.EDGE_SIZE:
                        break
                    line_points.append((px[0],px[1]+j))
            else:
                for j in xrange(thickness):
                    if px[0]+j>=self.EDGE_SIZE:
                        break
                    line_points.append((px[0]+j,px[1]))
        return line_points

def main():
    w = Wheel()
    w.calculate()

if __name__ == "__main__":
    main()
