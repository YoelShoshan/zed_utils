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

#example usage:
#imshowmult(img1,img2,img3,cmap1='gray',interpolation2='nearest',title1='banana')
def imshowmult(*args, **kwargs):
    images_num = len(args)
    if 0==images_num:
        raise Exception('No images provided!')

    #for a in args:
    #    print(a)
    #for k, v in kwargs.iteritems():
    #    print("%s = %s" % (k, v))

    figures_side = np.ceil(np.sqrt(images_num))
    fig = plt.figure()

    base_axis = None

    for i,img in enumerate(args):
        sharex = None
        sharey = None
        if base_axis:
            sharex = base_axis
            sharey = base_axis

        ax = fig.add_subplot(figures_side,figures_side, i+1, sharex=sharex, sharey=sharey)

        if not base_axis:
            base_axis = ax

        cmap_str = 'cmap%d'%i
        camp_val = None
        if cmap_str in kwargs:
            camp_val = kwargs[cmap_str]

        interpolation_str = 'interpolation%d' % i
        interpolation_val = None
        if interpolation_str in kwargs:
            interpolation_val = kwargs[interpolation_str]

        ax.imshow(img, cmap=camp_val, interpolation=interpolation_val)

    plt.show()

def test_imshowmult():
    import scipy.misc
    import os
    def compare(num):
        input_filename = '%06d_INPUT_PATCH.png' % num
        detection_filename = '%06d_PREDICTION_PATCH.png' % num
        gt_mask_filename = '%06d_GT_MASK.png' % num
        images_loc = '/gpfs/haifa/projects/m/msieve_dev2/usr/yoel/dev/dbs/MG/all_tumors_diseases/vis/prediction_fullimage'
        input_img = scipy.misc.imread(os.path.join(images_loc,input_filename))
        detection_img = scipy.misc.imread(os.path.join(images_loc,detection_filename))
        detection_img = scipy.misc.imresize(detection_img, (6000, 6000))
        gt_mask_img = scipy.misc.imread(os.path.join(images_loc,gt_mask_filename))
        imshowmult(input_img, detection_img, gt_mask_img, detection_img > 1, cmap0='gray',
                        interpolation1='nearest')
    compare(35)


#test_imshowmult()


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

