import png
import numpy as np
import matplotlib.pyplot as plt

def imshowpair(im1,im2, cmap1=None, cmap2=None):
    fig = plt.figure()
    ax0 = fig.add_subplot(121)
    if None!=cmap1:
        ax0.imshow(im1,cmap=cmap1)
    else:
        ax0.imshow(im1)
    ax1 = fig.add_subplot(122, sharex=ax0, sharey=ax0)
    if None != cmap2:
        ax1.imshow(im2, cmap=cmap2)
    else:
        ax1.imshow(im2)

    plt.show()

def savepng(filename,arr):
    if len(arr.shape) != 2:
        raise(ValueError('savepng only supports 2d numpy arrays'))

    bits_num = -1

    if arr.dtype == 'uint8':
        bits_num = 8
    elif arr.dtype == 'uint16':
        bits_num = 16
    else:
        raise(ValueError('savepng only supports uint8 and uint16! {0} is not supported.'.format(arr.dtype)))

    with open(filename, 'wb') as f:
        writer = png.Writer(width=arr.shape[1], height=arr.shape[0], bitdepth=bits_num, greyscale=True)
        writer.write(f, arr.tolist())

def to_rgb(im):
    im.resize((im.shape[0], im.shape[1], 1))
    return np.repeat(im.astype(im.dtype), 3, 2)

def simple_scale_to_uint8(im):
    #min_val = np.min(im)
    max_val = np.max(im)

    im = im*(255.0/max_val)
    return im.astype('uint8')

