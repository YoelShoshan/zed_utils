import numpy as np
from . import zed_affine_transformations as zat
import dicom
import cv2
import matplotlib.pyplot as plt
import scipy.misc

#each column in the created matrix is a point in the aabb
def AABBToPoints(aabb_top, aabb_bottom, aabb_left, aabb_right):
    #clockwise order
    points = np.array([
        [aabb_left, aabb_top, 1], #top left
        [aabb_left, aabb_bottom, 1],  # top right
        [aabb_right, aabb_bottom, 1],  # bottom right
        [aabb_right,aabb_top, 1],  # bottom left
        ]).T
    return points

def Debug_DrawAABB(img, aabb_top, aabb_bottom, aabb_left, aabb_right):
    aabb_top = int(aabb_top)
    aabb_bottom = int(aabb_bottom)
    aabb_left = int(aabb_left)
    aabb_right = int(aabb_right)
    cv2.line(img,(aabb_left,aabb_top),(aabb_right,aabb_top),(255))
    cv2.line(img, (aabb_left, aabb_bottom), (aabb_right, aabb_bottom), (255))
    cv2.line(img, (aabb_left, aabb_top), (aabb_left, aabb_bottom), (255))
    cv2.line(img, (aabb_right, aabb_top), (aabb_right, aabb_bottom), (255))


def resize_bboxes(bboxes, resize_by):
    resized_bboxes = []
    for b in bboxes:
        # resized_bboxes.append([b[0]//resize_by,b[1]//resize_by,b[2]//resize_by,b[3]//resize_by])
        resize_bb = [b[0] // resize_by, b[1] // resize_by, b[2] // resize_by, b[3] // resize_by]
        resized_bboxes.append(resize_bb)
    return resized_bboxes

def get_aabb_from_contour(X, Y):
    x_min = np.min(X)
    y_min = np.min(Y)
    x_max = np.max(X)
    y_max = np.max(Y)
    return (x_min, y_min, x_max, y_max)

def SubRectSafe(np_arr, row_start, row_end, col_start, col_end , verbose=False):
    #debug
    verbose=False

    if verbose:
        print('SubRectSafe input: ', row_start, row_end, col_start, col_end)

    if col_start>=0 and col_end<np_arr.shape[1] and row_start>=0 and row_end<np_arr.shape[0]:
        if verbose:
            print('return as is.')
        return np_arr[row_start:row_end+1,col_start:col_end+1]

    #we are out of bounds
    safe_arr = np.zeros((row_end-row_start+1,col_end-col_start+1), dtype=np_arr.dtype)
    safe_row_start = max(0,row_start)
    safe_row_end = min(np_arr.shape[0]-1,row_end+1)
    safe_col_start = max(0,col_start)
    safe_col_end = min(np_arr.shape[1]-1,col_end+1)

    if verbose:
        print('SubRectSafe output: ', safe_row_start, safe_row_end, safe_col_start, safe_col_end)

    safe_arr[:safe_row_end-safe_row_start,:safe_col_end-safe_col_start] = np_arr[safe_row_start:safe_row_end,safe_col_start:safe_col_end]

    return safe_arr

#normalizes the patch to be in -1,1 range.
#assumes input values in the range[0,4095]
def normalize_patch(patch):
    #patch_flat = patch.flatten()
    ####min_val = np.min(patch_flat)
    ####if min_val >0:
    ####    patch_flat -= np.min(patch_flat)
    #patch_flat = patch_flat.astype('float64') / 4095.0#np.max(patch_flat)
    #patch_flat = (patch_flat*2.0)-1.0
    #return patch_flat.reshape(patch.shape)

    patch = patch.astype(np.float32) / 4095.0
    patch *= 2.0
    patch -= 1.0

    return patch

#order for pertrub_color is multiply then add
def TransformPatch(img,normalize, pertrub_color_ADD, pertrub_color_MULTIPLY, rot_ang, flip_x, flip_y, aabb_top, aabb_bottom, aabb_left, aabb_right, interpolation_method=cv2.INTER_LANCZOS4):
    #print('dbg - remove!')
    #Debug_DrawAABB(img, aabb_top, aabb_bottom, aabb_left, aabb_right)

    #print('k1')

    (all_pos_top, all_pos_bottom, all_pos_left, all_pos_right) = GetAllPossibleRotationsAABB(aabb_top, aabb_bottom, aabb_left, aabb_right)

    #print('k2')

    #orig - not safe for out of bounds
    #src_img_patch = img[all_pos_top:all_pos_bottom+1, all_pos_left:all_pos_right+1]

    src_img_patch = SubRectSafe(img,all_pos_top,all_pos_bottom+1, all_pos_left,all_pos_right+1)

    #print('k3')

    Mrot = zat.GenRotateAroundPoint2D(src_img_patch.shape[1]/2, src_img_patch.shape[0]/2,rot_ang)
    Mflip = zat.GenFlip2D(flip_x, flip_y, src_img_patch.shape[1], src_img_patch.shape[0])
    M = np.dot(Mrot,Mflip)

    #print('k4')

    #TODO: sould we replicate or reflect? or maybe make it a param?
    #TODO: should we use the WARP_INVERSE_MAP ? can result in better looking images
    #TODO: think about transformations to polar space
    #warpAffine accepts 3x2 and not 3x3

    if src_img_patch.dtype==np.bool: #warpAffine doesn't handle bool type
        src_img_patch = src_img_patch.astype('uint8')

    #print('k5')

    out_img = cv2.warpAffine(src_img_patch, M[:2,:], src_img_patch.shape, borderMode=cv2.BORDER_REFLECT, flags=interpolation_method)

    if src_img_patch.dtype == np.bool:
        out_img = out_img.astype(np.bool)

    #print('k6')

    center_x = src_img_patch.shape[1]/2
    center_y = src_img_patch.shape[0]/2
    #out_rad = max(aabb_top-aabb_bottom,aabb_right-aabb_left)

    x_rad = (aabb_right-aabb_left)/2
    y_rad = (aabb_bottom - aabb_top)/2

    out_img_patch = out_img[int(center_y-y_rad):int(center_y+y_rad), int(center_x-x_rad):int(center_x+x_rad)]
    #print('k7')

    if normalize:
        out_img_patch = normalize_patch(out_img_patch)

    need_clipping = False

    if None!=pertrub_color_ADD and pertrub_color_ADD != 0.0:
        out_img_patch += pertrub_color_ADD
        need_clipping = True

    if None!=pertrub_color_MULTIPLY and pertrub_color_MULTIPLY != 0.0:
        out_img_patch *= pertrub_color_MULTIPLY
        need_clipping = True

    if need_clipping:
        out_img_patch.clip(-1.0,1.0,out_img_patch)

    if 250 != out_img_patch.shape[0] or 250 != out_img_patch.shape[1]:
        z = 123

    #print('k8')

    return out_img_patch


INTERP_IND_TO_NAME = {0:cv2.INTER_LINEAR, 1:cv2.INTER_CUBIC, 2:cv2.INTER_AREA, 3:cv2.INTER_LANCZOS4}

def GetRandomTransformationVars(color_add_range, color_mult_range, angle_range, flip_x, flip_y, offset_x_range, offset_y_range,aabb_top, aabb_bottom, aabb_left, aabb_right):

    if color_add_range>0.0:
        color_add = np.random.uniform(-color_add_range, color_add_range)
    else:
        color_add = 0.0

    if color_mult_range>0.0:
        color_mult = 1.0+np.random.uniform(-color_mult_range, color_mult_range)
    else:
        color_mult = 0.0


    ang = np.random.uniform(-angle_range, angle_range)
    if flip_x:
        flip_x_val = np.random.randint(0, 2)
    else:
        flip_x_val = 0

    if flip_y:
        flip_y_val = np.random.randint(0, 2)
    else:
        flip_y_val = 0
    interp_ind = np.random.randint(0, len(INTERP_IND_TO_NAME))

    offset_x = int(np.random.uniform(-offset_x_range, offset_x_range))
    offset_y = int(np.random.uniform(-offset_y_range, offset_y_range))

    aabb_left += offset_x
    aabb_left = int(aabb_left)
    aabb_right += offset_x
    aabb_right = int(aabb_right)
    aabb_top += offset_y
    aabb_top = int(aabb_top)
    aabb_bottom += offset_y
    aabb_bottom = int(aabb_bottom)


    return color_add, color_mult, ang, flip_x_val, flip_y_val, interp_ind, offset_x, offset_y, aabb_top, aabb_bottom, aabb_left, aabb_right

def RandomTransformPatch(img,angle_range, flip_x, flip_y, offset_x_range, offset_y_range, aabb_top, aabb_bottom, aabb_left, aabb_right):
    ang, flip_x_val, flip_y_val, interp_ind, offset_x, offset_y, aabb_top, aabb_bottom, aabb_left, aabb_right = GetRandomTransformationVars(angle_range, flip_x, flip_y, offset_x_range, offset_y_range)
    rand_patch = TransformPatch(img,ang,flip_x_val, flip_y_val, aabb_top, aabb_bottom, aabb_left, aabb_right, interpolation_method=INTERP_IND_TO_NAME[interp_ind])
    return rand_patch


def GetAllPossibleRotationsAABB(aabb_top, aabb_bottom, aabb_left, aabb_right):
    center_x = (aabb_left+aabb_right)/2
    center_y = (aabb_top+aabb_bottom)/2
    top_left_x = aabb_left
    top_left_y = aabb_top

    #calculate length of the aabb diagonal
    diag_length = np.sqrt( ((center_x-top_left_x)**2) + ((center_y-top_left_y)**2))
    out_top = center_y-diag_length
    out_bottom = center_y+diag_length
    out_left = center_x-diag_length
    out_right = center_x+diag_length

    return (int(out_top),int(out_bottom), int(out_left), int(out_right))

def random_coord_2d_int(img,aabb_size):
    half_size = int((aabb_size//2)-3)
    R = np.random.randint(half_size,img.shape[0]-half_size)
    C = np.random.randint(half_size,img.shape[1]-half_size)
    return (R,C)

def random_non_empty_coord_2d(img,aabb_size):
    for i in range(1000):
        R,C = random_coord_2d_int(img,aabb_size)
        if img[R,C] > 0:
            return R, C
    raise ValueError('could not get a non empty random point in many attempts! Is this a black image?')

def RandomAugmentedPatch(img, aabb_size, angle_range, flip_x, flip_y, offset_x_range, offset_y_range):
    (mid_r, mid_c) = random_non_empty_coord_2d(img)

    aabb_top = mid_r - aabb_size
    aabb_bottom = mid_r + aabb_size
    aabb_left = mid_c - aabb_size
    aabb_right = mid_c + aabb_size
    #new_patch = RandomTransformPatch(img, np.pi * 2, True, True, 50.0, 50.0, aabb_top, aabb_bottom, aabb_left, aabb_right)
    new_patch = RandomTransformPatch(img, angle_range, flip_x, flip_y, offset_x_range, offset_y_range, aabb_top, aabb_bottom, aabb_left, aabb_right)

    return new_patch

def test_random_generate_patches(centers_iterations, per_center_iterations):
    np.random.seed(1337) #reproducable (change when making ensembles)
    dc = dicom.read_file(r"C:\dev\dbs\MG\all_tumors_diseases\copied_local\train\00357@view@@_SZ_SZ612_20130218_160955_MG_8_bit_SZ612_4.Ser9.Img0.dcm_8bit.dcm")
    img = dc.pixel_array
    img[img == 4095] = 0  # remove overlay
    img = (img.astype('float32') * (255.0 / 4095.0)).astype('uint8')
    aabb_size = 200
    #non_zero_indices = img>0
    #note: remember that in practice it's faster to do a permutation (and you also make sure you don't repeat points too much)
    #      it also allows you to unravel many flat indices in one go
    for c in range(per_center_iterations):
        if 0==c%10:
            print(c)
        #(mid_r,mid_c) = random_coord_2d(img)
        (mid_r, mid_c) = random_non_empty_coord_2d(img)

        aabb_top = mid_r-aabb_size
        aabb_bottom = mid_r+aabb_size
        aabb_left = mid_c-aabb_size
        aabb_right = mid_c+aabb_size
        for i in range(centers_iterations):
            new_patch = RandomTransformPatch(img, np.pi*2, True, True, 50.0, 50.0, aabb_top, aabb_bottom, aabb_left, aabb_right)
            #not using scipy.misc.imsave() because it will normalize
            scipy.misc.toimage(new_patch, cmin=0, cmax=255).save(r'C:\dev\dbs\MG\deep_tumor\visualization\augmentation\%06d_%06d_%.2f_%.2f_aug_%d.png' % (c,i,mid_c, mid_r,i))






def test():
    dc = dicom.read_file(r"C:\dev\dbs\MG\all_tumors_diseases\copied_local\train\00357@view@@_SZ_SZ612_20130218_160955_MG_8_bit_SZ612_4.Ser9.Img0.dcm_8bit.dcm")
    img = dc.pixel_array
    img[img == 4095] = 0  # remove overlay
    img = (img.astype('float32') * (255.0/ 4095.0)).astype('uint8')
    R, C = img.shape

    aabb_left = 1350
    aabb_right = 1500
    aabb_top = 1800
    aabb_bottom = 2000

    #Debug_DrawAABB(img, aabb_top, aabb_bottom, aabb_left, aabb_right)

    all_pos_ang_aabb = GetAllPossibleRotationsAABB(aabb_top, aabb_bottom, aabb_left, aabb_right)
    Debug_DrawAABB(img, *all_pos_ang_aabb)

    if False:
        points = AABBToPoints(aabb_top, aabb_bottom, aabb_left, aabb_right)

        def try_ang(ang):
            M = zat.GenRotateAroundPoint2D((aabb_left + aabb_right) / 2, (aabb_top + aabb_bottom) / 2, ang)
            trans_points = np.dot(M, points)

            for i in range(4):
                next = (i + 1) % 4
                cv2.line(img, (int(trans_points[0, i]), int(trans_points[1, i])), (int(trans_points[0, next]), int(trans_points[1, next])), (255))

                # for a in np.arange(0.0,np.pi*2,np.pi/180.0):

        for a in np.arange(0.0, np.pi * 2, (np.pi / 180.0) * 10):
            try_ang(a)

    def TestFlip(ang):
        plt.figure(1)
        plt.imshow(TransformPatch(img, (np.pi / 180.0) * ang, False, False, aabb_top, aabb_bottom, aabb_left, aabb_right))
        plt.figure(2)
        plt.imshow(TransformPatch(img, (np.pi / 180.0) * ang, True, False, aabb_top, aabb_bottom, aabb_left, aabb_right))
        plt.figure(3)
        plt.imshow(TransformPatch(img, (np.pi / 180.0) * ang, False, True, aabb_top, aabb_bottom, aabb_left, aabb_right))
        plt.figure(4)
        plt.imshow(TransformPatch(img, (np.pi / 180.0) * ang, True, True, aabb_top, aabb_bottom, aabb_left, aabb_right))

    a = 123

    #try out of boundz
    plt.imshow(RandomTransformPatch(img, 0.0, False, False, 0.0, 0.0, 3200, 3500,2500, 2800))

    plt.imshow(RandomTransformPatch(img, 0.0, False, False, 0.0, 0.0, aabb_top, aabb_bottom, aabb_left, aabb_right))

    print('done.')
    plt.show()


#test()
#test_random_generate_patches(50,50)