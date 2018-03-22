import glob
import sys
import sep
import numpy as np
import transinet_funcs as tfuncs
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt 
import commands


def run_sep(sub):
    ''' returns coordinates of bright objects in
    subtraction
    '''

    x_coords = []
    y_coords = []
    extract = sep.extract(sub, thresh=0.035)
    for obj in extract:
        x = int(obj[7])
        y = int(obj[8])
        if np.isnan(np.sum(sub[y-20:y+21, x-20:x+21])):
            continue
        else:
            x_coords.append(x)
            y_coords.append(y)

    coords = np.array([x_coords, y_coords])
    coords = coords.T
    return coords


def main():
    ''' steps:
    1) load images
    2) subtract images
    3) run sep to find bright objects
    4) make postage stamps around sep objects
    '''
    set_num = sys.argv[1]
    path = '/Users/mcurrie/Projects/TransiNet/data/set_%s_epochs/' % set_num
    filter_epochs = glob.glob(path + 'F*drz*')
    filters = []
    epochs = []
    for im in filter_epochs:
        im_no_path = im.split('/')[-1]
        filt, epoch, junk = im_no_path.split('_')
        filters.append(filt)
        epochs.append(epoch)
    unique_filts = np.unique(filters)
    unique_epochs = np.unique(epochs)
    assert len(unique_epochs) == 2
    subtractions = {}
    for filt in unique_filts:
        subtractions[filt] = []
        im1 = pyfits.open(path + '%s_%s_drz.fits' % (filt, unique_epochs[0]))
        im2 = pyfits.open(path + '%s_%s_drz.fits' % (filt, unique_epochs[1]))
        data1 = im1[1].data
        data2 = im2[1].data
        im1.close()
        im2.close()
        sub1 = data1 - data2
        sub2 = data2 - data1
        subtractions[filt].append(sub1)
        subtractions[filt].append(sub2)

    for key in subtractions.keys():
        subtractions[key] = np.array(subtractions[key])

    obj_coords1 = np.empty((0, 2))
    obj_coords2 = np.empty((0, 2))

    for filt in subtractions.keys():
        # run sep on subtraction:
        obj_coords1 = np.concatenate((obj_coords1,
                                      run_sep(subtractions[filt][0])))
        obj_coords2 = np.concatenate((obj_coords2,
                                      run_sep(subtractions[filt][1])))

    obj_coords1 = np.unique(obj_coords1, axis=0)
    obj_coords2 = np.unique(obj_coords2, axis=0)
    stamp_size = 14

    num_filters = len(unique_filts)
    all_stamps = np.empty((0, stamp_size*2, stamp_size*2, num_filters))
    info = []
    c = 0
    for coords in obj_coords1:
        x, y = coords
        filts = subtractions.keys()
        sub1 = subtractions[filts[0]][0]
        #pyfits.writeto('sub_test.fits', sub1)
        #assert False
        sub2 = subtractions[filts[1]][0]
        stamp1 = tfuncs.make_stamp(int(x), int(y), sub1, stamp_size)
        stamp2 = tfuncs.make_stamp(int(x), int(y), sub2, stamp_size)
        try:
            stamp1_temp = stamp1 + np.abs(np.nanmin(stamp1))
            stamp2_temp = stamp2 + np.abs(np.nanmin(stamp2))
            stamp1_temp /= np.nanmax(stamp1_temp)
            stamp2_temp /= np.nanmax(stamp2_temp)
            obj_stamps = np.stack((stamp1_temp, stamp2_temp), axis=-1)
            all_stamps = np.append(all_stamps, [obj_stamps], axis=0)
            info.append([x,y])
        except:
            pass
    for coords in obj_coords2:
        x, y = coords
        filts = subtractions.keys()
        sub1 = subtractions[filts[0]][1]
        sub2 = subtractions[filts[1]][1]
        stamp1 = tfuncs.make_stamp(int(x), int(y), sub1, stamp_size)
        stamp2 = tfuncs.make_stamp(int(x), int(y), sub2, stamp_size)
        try:
            stamp1_temp = stamp1 + np.abs(np.nanmin(stamp1))
            stamp2_temp = stamp2 + np.abs(np.nanmin(stamp2))
            stamp1_temp /= np.nanmax(stamp1_temp)
            stamp2_temp /= np.nanmax(stamp2_temp)
            obj_stamps = np.stack((stamp1_temp, stamp2_temp), axis=-1)
            all_stamps = np.append(all_stamps, [obj_stamps], axis=0)
            info.append([x,y])
        except:
            pass
    info = np.array(info, dtype=np.int)
    n, x, y, z = all_stamps.shape
    print 'Found', n, 'candidates'
    np.save('../../data/candidate_info_set_%s.npy' % set_num, info)
    np.save('../../data/object_candidates_set_%s.npy' % set_num, all_stamps)


main()
