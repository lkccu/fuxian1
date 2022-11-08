import canny_edge_detector_3D as ced
import slicer_3D
import numpy as np
import matplotlib.pyplot as plt
from medpy.io.load import load

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img = load(r'E:\course\learn2reg\task1\dataset\TrainA\img0062_bcv_CT.nii.gz')
img = img[0]

detector = ced.cannyEdgeDetector3D(img, sigma=0.6, lowthresholdratio=0.3, highthresholdratio=0.2, weak_voxel=75, strong_voxel=255)
img_edges = detector.detect()


fig, (ax1,ax2) = plt.subplots(1, 2)
plt.subplots_adjust(wspace=0.3)

tracker1 = slicer_3D.Slicer3D(ax1, img_edges)
tracker2 = slicer_3D.Slicer3D(ax2, img)

fig.canvas.mpl_connect('scroll_event', tracker1.on_scroll)
fig.canvas.mpl_connect('scroll_event', tracker2.on_scroll)

fig.set_figheight(8)
fig.set_figwidth(8)
plt.show()