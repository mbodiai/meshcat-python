from __future__ import absolute_import, division, print_function


import numpy as np


from meshcat.geometry import Points, PointsGeometry, PointsMaterial
from meshcat.visualizer import Visualizer
verts = np.random.random((3, 100000)).astype(np.float32)

vis = Visualizer().open()
vis.set_object(Points(
    PointsGeometry(verts, color=verts),
    PointsMaterial(size=0.001)
))
