import numpy as np
import matplotlib.pyplot as plt
import sep
import astropy.io.fits as pyfits
import sys
set_nums = np.arange(0, 25, 1, dtype=int)
#set_nums = ['play3']
run_sep = 1 
training_data = 0




def run_sep_on_sub(bad_pix_dir, set_num):
    path = '/Users/mcurrie/GitRepos/step_1/%s/set_%s_epochs/' % (bad_pix_dir,
                                                              str(set_num))
    try:
        data_fl = path + 'F125W_epoch02_drz.fits'
        ref_fl = path + 'F125W_epoch01_drz.fits'
        data = pyfits.open(data_fl)
        ref = pyfits.open(ref_fl)
    except IOError:
        data_fl = path + 'F110W_epoch02_drz.fits'
        ref_fl = path + 'F110W_epoch01_drz.fits'
        data = pyfits.open(data_fl)
        ref = pyfits.open(ref_fl)


    sub = data[1].data - ref[1].data
    mask = (data[3].data > 0) & (ref[3].data > 0)
    mask = ~mask
    data.close()
    ref.close()
    extract = sep.extract(sub, mask=mask, thresh=0.035)

    with open('/Users/mcurrie/GitRepos/step_1/%s/object_coords_table.txt'
              % bad_pix_dir, 'wb') as fl:
        fl.write('x\ty\n')
        for obj in extract:
            x = int(obj[7])
            y = int(obj[8])
            if np.isnan(np.sum(sub[y-20:y+21, x-20:x+21])):
                print 'found a nan'
                continue
            else:
                fl.write('%i\t%i\n' % (x, y))


def get_mags(bad_pix_dir, set_num):
    path = '/Users/mcurrie/GitRepos/step_1/%s/sn_files/' % bad_pix_dir
    fl = path + 'simulated_set_%s.cat' % str(set_num)
    mag_list = []
    with open(fl, 'rb') as f:
        for line in f:
            if line[0] == '#':
                continue
            else:
                line = line.split(None)
                mag = line[2]
                mag_list.append(mag)
    return np.array(mag_list, dtype=np.float32)


def get_coords(bad_pix_dir, set_num):
    path = '/Users/mcurrie/GitRepos/step_1/%s/sn_files/' % bad_pix_dir
    fl = path + 'ds9_set_%s.reg' % str(set_num)
    x_list = []
    y_list = []
    with open(fl, 'rb') as f:
        for line in f:
            line = line.split('(')
            if len(line) == 2:
                line = line[1]
                line = line.split(',')
                x = line[0]
                y = line[1]
                x_list.append(int(x))
                y_list.append(int(y))

            else:
                continue
    return np.array(x_list), np.array(y_list)


def make_ps(data, x, y, ps_size):
    ps = data[y-ps_size : y+ps_size+2,
              x-ps_size : x+ps_size+2]

    return ps

def normalize_ps(ps):

    ps = ps + np.abs(np.nanmin(ps))

    ps = ps/np.nanmax(ps)

    return ps

badpix = sys.argv[1]
if badpix == 'new':
    badpix_dir = '/sets_newbadpix/'
elif badpix == 'old':
    badpix_dir = '/sets_oldbadpix/'
else:
    assert False

num_channels = 3
im_size = 16
ps_stack_master = np.empty((0, 32, 32, num_channels))
labels_master = np.empty(0)
info_master = np.empty((0, 3))
mags_master = np.empty(0)
for set_num in set_nums:
    print 'WORKING ON SET', set_num
    mags = get_mags(badpix_dir, set_num)
    x, y = get_coords(badpix_dir, set_num)
    labels = []
    for mag in mags:
        if mag == 32.:
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels, dtype=np.int)
    mags = np.array(mags)
    path = '/Users/mcurrie/GitRepos/step_1/%s/set_%s_epochs/' % (badpix_dir,
                                                                 str(set_num))
    data_fl = path + 'F160W_epoch02_drz.fits'
    ref_fl = path + 'F160W_epoch01_drz.fits'
    
    data = pyfits.open(data_fl)
    ref = pyfits.open(ref_fl)

    sub = data[1].data - ref[1].data
    '''
    for n in range(len(x))[:20]:
        ps = sub[y[n]-10:y[n]+11, x[n]-10:x[n]+11]
        plt.figure()
        plt.imshow(ps)
        plt.title(labels[n])
    plt.show()
    assert False
    '''
    if run_sep:

        run_sep_on_sub(badpix_dir, set_num)

        x_artifacts, y_artifacts = \
        np.genfromtxt('/Users/mcurrie/GitRepos/step_1/%s/object_coords_table.txt'
                      % badpix_dir,
                                                 skip_header=1,
                                                 dtype=np.int).T
        with open('test.txt', 'wb') as f:
            for item in zip(x_artifacts, y_artifacts):
                f.write(str(item) + '\n')
            f.write('\n')
            for item in zip(x, y):
                f.write(str(item) + '\n')
        print zip(x_artifacts, y_artifacts)
        print zip(x, y)
        counter = 0
        match_args = []
        for n in range(len(x_artifacts)):
            matches_for_this = 0
            for m in range(len(x)):
                if np.isclose(x_artifacts[n], x[m], atol=3) and \
                   np.isclose(y_artifacts[n], y[m], atol=3):
                    print '\nfound a close match for arg %i in SN list:' % n
                    print 'x:',x_artifacts[n], x[m]
                    print 'y:',y_artifacts[n], y[m]
                    match_args.append(n)
                    counter+=1
                    matches_for_this+=1
                    print counter
            assert matches_for_this < 2
        match_args = np.array(match_args)
        new_artifacts_bool = np.ones(len(x_artifacts))
        new_artifacts_bool[match_args] = 0
        new_artifacts_bool = new_artifacts_bool.astype(bool)
        new_labels = np.ones_like(x_artifacts[new_artifacts_bool])
        new_mags = 32*np.ones_like(x_artifacts[new_artifacts_bool])
        x = np.concatenate((x, x_artifacts[new_artifacts_bool]))
        y = np.concatenate((y, y_artifacts[new_artifacts_bool]))
        labels = np.concatenate((labels, new_labels))
        mags = np.concatenate((mags, new_mags))
        #print mags
        #print np.unique(mags)
        #assert np.unique(mags)[0] > 0

    data_path = '/Users/mcurrie/GitRepos/step_1/%s/set_%s_epochs/'%(badpix_dir,
                                                                    str(set_num))

    try:
        SN_im_fl_1 = data_path + 'F125W_epoch02_drz.fits'
        ref_im_fl_1 = data_path + 'F125W_epoch01_drz.fits'
        SN_im_1 = pyfits.open(SN_im_fl_1)
        ref_im_1 = pyfits.open(ref_im_fl_1)
    except IOError:
        SN_im_fl_1 = data_path + 'F110W_epoch02_drz.fits'
        ref_im_fl_1 = data_path + 'F110W_epoch01_drz.fits'
        SN_im_1 = pyfits.open(SN_im_fl_1)
        ref_im_1 = pyfits.open(ref_im_fl_1)

    SN_im_fl_2 = data_path + 'F160W_epoch02_drz.fits'
    ref_im_fl_2 = data_path + 'F160W_epoch01_drz.fits'


    SN_data_1 = SN_im_1[1].data
    ref_data_1 = ref_im_1[1].data
    SN_im_2 = pyfits.open(SN_im_fl_2)
    SN_data_2 = SN_im_2[1].data
    ref_im_2 = pyfits.open(ref_im_fl_2)
    ref_data_2 = ref_im_2[1].data


    sub_data_1 = SN_data_1 - ref_data_1
    sub_data_2 = SN_data_2 - ref_data_2
    ps_size = im_size/2 - 1

    ps_stack = np.empty((0, 32, 32, num_channels))
    new_labels = []
    new_mags = []
    info = []
    for n in range(len(x)):

        ps_im1 = make_ps(SN_data_1, x[n], y[n], ps_size)
        ps_ref1 = make_ps(ref_data_1, x[n], y[n], ps_size)
        ps_sub1 = make_ps(sub_data_1, x[n], y[n], ps_size)
        ps_im2 = make_ps(SN_data_2, x[n], y[n], ps_size)
        ps_ref2 = make_ps(ref_data_2, x[n], y[n], ps_size)
        ps_sub2 = make_ps(sub_data_2, x[n], y[n], ps_size)

        ps_im1 = np.pad(ps_im1, pad_width=8, mode='constant')
        ps_ref1 = np.pad(ps_ref1, pad_width=8, mode='constant')
        ps_sub1 = np.pad(ps_sub1, pad_width=8, mode='constant')
        ps_im2 = np.pad(ps_im2, pad_width=8, mode='constant')
        ps_ref2 = np.pad(ps_ref2, pad_width=8, mode='constant')
        ps_sub2 = np.pad(ps_sub2, pad_width=8, mode='constant')

        ps_sub1 = np.clip(ps_sub1, -0.5, 3)
        ps_sub2 = np.clip(ps_sub2, -0.5, 3)
        '''
        ps_1 = sub_data_1[y[n]-ps_size : y[n]+ps_size+2,
                          x[n]-ps_size : x[n]+ps_size+2]
        ps_2 = sub_data_2[y[n]-ps_size : y[n]+ps_size+2,
                          x[n]-ps_size : x[n]+ps_size+2]
        ps_1 = np.clip(ps_1, -0.5, 2)
        ps_2 = np.clip(ps_2, -0.5, 2)
        '''
        if np.isnan(np.sum(ps_sub1)) or np.isnan(np.sum(ps_sub2)):
            pass
        elif ps_sub1.shape != (32,32):
            pass
        elif ps_sub2.shape != (32,32):
            pass
        else:
            ps_im1 = normalize_ps(ps_im1)
            ps_ref1 = normalize_ps(ps_ref1)
            ps_sub1 = normalize_ps(ps_sub1)
            ps_im2 = normalize_ps(ps_im2)
            ps_ref2 = normalize_ps(ps_ref2)
            ps_sub2 = normalize_ps(ps_sub2)
            '''
            ps_sub1 = ps_sub1 + np.abs(np.nanmin(ps_sub1))
            ps_sub2 = ps_sub2 + np.abs(np.nanmin(ps_sub2))

            ps_1 = ps_1/np.nanmax(ps_1)
            ps_2 = ps_2/np.nanmax(ps_2)
            '''
            new_labels.append(labels[n])
            new_mags.append(mags[n])
            if num_channels == 6:
                temp_stack = np.stack((ps_im1,
                                       ps_ref1,
                                       ps_sub1,
                                       ps_im2,
                                       ps_ref2,
                                       ps_sub2), axis=-1)
            elif num_channels == 2:
                temp_stack = np.stack((ps_sub1,
                                       ps_sub2), axis=-1)
            elif num_channels == 3:
                temp_stack = np.stack((ps_sub1,
                                       ps_sub2,
                                       0.5*(ps_sub1+ps_sub2)),
                                       axis=-1)
            else:
                print 'Invalid number of channels'
                assert False

            ps_stack = np.append(ps_stack, [temp_stack], axis=0)
            info.append([int(set_num), x[n], y[n]])

    ps_stack = ps_stack.astype(np.float32)
    new_labels = np.array(new_labels, dtype=np.int)
    new_mags = np.array(new_mags)
    info = np.array(info)

    for ps in ps_stack[:,:,:,0]:
        if np.isnan(np.sum(ps)):
            print ps
            assert False
    

    ps_stack_master = np.concatenate((ps_stack_master, ps_stack))
    labels_master = np.concatenate((labels_master, new_labels))
    mags_master = np.concatenate((mags_master, new_mags))
    info_master = np.concatenate((info_master, info))
print ps_stack_master.shape
print labels_master.shape
print info_master.shape
print mags_master.shape
mags_master = np.around(mags_master, 2)
print mags_master
'''
for n in range(20):
    plt.figure()
    plt.imshow(ps_stack_master[n, :, :, 0])
    plt.colorbar()
    plt.title(str(labels_master[n]) + ' ' + str(mags_master[n]))

plt.show()
'''
ps_stack_master_new = np.empty((0, 32, 32, num_channels))
labels_master_new = np.empty(0)
info_master_new = np.empty((0, 3))
mags_master_new = np.empty(0)

labels_master_new = np.concatenate((labels_master_new, labels_master))
labels_master_new = np.concatenate((labels_master_new, labels_master))
labels_master_new = np.concatenate((labels_master_new, labels_master))
labels_master_new = np.concatenate((labels_master_new, labels_master))

info_master_new = np.concatenate((info_master_new, info_master))
info_master_new = np.concatenate((info_master_new, info_master))
info_master_new = np.concatenate((info_master_new, info_master))
info_master_new = np.concatenate((info_master_new, info_master))

mags_master_new = np.concatenate((mags_master_new, mags_master))
mags_master_new = np.concatenate((mags_master_new, mags_master))
mags_master_new = np.concatenate((mags_master_new, mags_master))
mags_master_new = np.concatenate((mags_master_new, mags_master))

ps_stack_master_new = np.concatenate((ps_stack_master_new,
                                      np.rot90(ps_stack_master, k=1,
                                               axes=(1, 2))))
ps_stack_master_new = np.concatenate((ps_stack_master_new,
                                      np.rot90(ps_stack_master, k=2,
                                               axes=(1, 2))))
ps_stack_master_new = np.concatenate((ps_stack_master_new,
                                      np.rot90(ps_stack_master, k=3,
                                               axes=(1, 2))))
ps_stack_master_new = np.concatenate((ps_stack_master_new,
                                      np.rot90(ps_stack_master, k=4,
                                               axes=(1, 2))))

print ps_stack_master_new.shape
print labels_master_new.shape
print info_master_new.shape
print mags_master_new.shape
np.save('/Users/mcurrie/GitRepos/TransiNet/data/%s/training_data.npy'
        % badpix_dir,
        ps_stack_master_new)
np.save('/Users/mcurrie/GitRepos/TransiNet/data/%s/training_labels.npy'
        % badpix_dir,
        labels_master_new)
np.save('/Users/mcurrie/GitRepos/TransiNet/data/%s/training_info.npy'
        % badpix_dir,
        info_master_new)
np.save('/Users/mcurrie/GitRepos/TransiNet/data/%s/training_mags.npy'
        % badpix_dir,
        mags_master_new)
