from __future__ import absolute_import, division, print_function

import math
import time

from meshcat.geometry import Box
from meshcat.transformations import rotation_matrix
from meshcat.visualizer import Visualizer


vis = Visualizer().open()

box = Box([0.5, 0.5, 0.5])
vis.set_object(box)

draw_times = []

vis["/Background"].set_property("top_color", [1, 0, 0])

for i in range(200):
    theta = (i + 1) / 100 * 2 * math.pi
    now = time.time()
    vis.set_transform(rotation_matrix(theta, [0, 0, 1]))
    draw_times.append(time.time() - now)
    time.sleep(0.01)

print(sum(draw_times) / len(draw_times))


