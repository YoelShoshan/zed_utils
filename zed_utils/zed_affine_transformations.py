import dicom
import cv2
import matplotlib.pyplot as plt
import numpy as np
from . import zed_image
import time

def GenRotationMat2D(ang):
    M = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang) , np.cos(ang) ,0], [0 ,0, 1]],dtype='float32')
    return M

def GenTranslate2D(x_off,y_off):
    M = np.array([[1, 0, x_off],[0,1,y_off],[0,0,1]],dtype='float32')
    return M

def GenFlip2D(flip_x, flip_y, size_x, size_y):
    flip_x_val = -1 if flip_x else 1
    flip_y_val = -1 if flip_y else 1
    shift_x = size_x if flip_x else 0.0
    shift_y = size_y if flip_y else 0.0

    M = np.array([[flip_x_val, 0, shift_x], [0, flip_y_val, shift_y], [0, 0, 1]], dtype='float32')
    return M


def GenRotateAroundPoint2D(x_off, y_off, ang):
    M = GenTranslate2D(x_off, y_off)
    M = np.dot(M,GenRotationMat2D(ang))
    M = np.dot(M, GenTranslate2D(-x_off, -y_off))
    return M



