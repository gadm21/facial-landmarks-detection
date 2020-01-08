
import numpy as np

#boundaries of a color to consider as blue
blue_lower= np.array([100, 60, 60])
blue_upper= np.array([140, 255, 255])

#kernel for erosion and dilation
kernel= np.ones((5,5), np.uint8)