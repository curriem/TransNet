import numpy as np
import matplotlib.pyplot as plt
import sep
import astropy.io.fits as pyfits

set_nums = np.arange(0, 25, 1, dtype=int)
#set_nums = ['play3']
run_sep = 1 
training_data = 0


def run_sep_on_sub(set_num):
    path = '/Users/mcurrie/GitRepos/step_1/set_%s_epochs/' % str(set_num)
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
    data.close()
    ref.close()
    extract = sep.extract(sub, thresh=0.035)

    with open('object_coords_table.txt', 'wb') as fl:
        fl.write('x\ty\n')
        for obj in extract:
            x = int(obj[7])
            y = int(obj[8])
            if np.isnan(np.sum(sub[y-20:y+21, x-20:x+21])):
                print 'found a nan'
                continue
            else:
                fl.write('%i\t%i\n' % (x, y))


def get_mags(set_num):
    path = '/Users/mcurrie/GitRepos/step_1/sn_files/'
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


def get_coords(set_num):
    path = '/Users/mcurrie/GitRepos/step_1/sn_files/'
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



ps_stack_master = np.empty((0, 32, 32, 2))
labels_master = np.empty(0)
info_master = np.empty((0, 3))
mags_master = np.empty(0)
for set_num in set_nums:
    print 'WORKING ON SET', set_num
    mags = get_mags(set_num)
    x, y = get_coords(set_num)

    labels = []
    for mag in mags:
        if mag == 32.:
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels, dtype=np.int)
    mags = np.array(mags)
    path = '/Users/mcurrie/GitRepos/step_1/set_%s_epochs/' % str(set_num)
    data_fl = path + 'F160W_epoch02_drz.fits'
    ref_fl = path + 'F160W_epoch01_drz.fits'
    
    data = pyfits.open(data_fl)
    ref = pyfits.open(ref_fl)

    sub = data[1].data - ref[1].data
#    sub_save = data
    #data.close()
    #ref.close()
 #   sub_save[1].data = sub

  #  sub_save.writeto('sub_test.fits')
   # assert False
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

        run_sep_on_sub(set_num)

        x_artifacts, y_artifacts = np.genfromtxt('object_coords_table.txt',
                                                 skip_header=1,
                                                 dtype=np.int).T
        counter = 0
        match_args = []
        for n in range(len(x_artifacts)):
            for m in range(len(x)):
                if np.isclose(x_artifacts[n], x[m], atol=3) and \
                   np.isclose(y_artifacts[n], y[m], atol=3):
                    print '\nfound a close match for arg %i in SN list:' % n
                    print 'x:',x_artifacts[n], x[m]
                    print 'y:',y_artifacts[n], y[m]
                    match_args.append(n)
                    counter+=1
                    print counter
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


    data_path = '/Users/mcurrie/GitRepos/step_1/set_%s_epochs/' % str(set_num)

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
    ps_size = 15

    ps_stack = np.empty((0, 32, 32, 2))
    new_labels = []
    new_mags = []
    info = []
    for n in range(len(x)):

        ps_1 = sub_data_1[y[n]-ps_size : y[n]+ps_size+2,
                          x[n]-ps_size : x[n]+ps_size+2]
        ps_2 = sub_data_2[y[n]-ps_size : y[n]+ps_size+2,
                          x[n]-ps_size : x[n]+ps_size+2]

        if np.isnan(np.sum(ps_1)) or np.isnan(np.sum(ps_2)):
            pass
        elif ps_1.shape != (32,32):
            pass
        elif ps_2.shape != (32,32):
            pass
        else:
            #ps = ps.flatten()
            try:
                ps_1 = ps_1 + np.abs(np.nanmin(ps_1))
            except:
                print ps_1.shape
                assert False
            ps_2 = ps_2 + np.abs(np.nanmin(ps_2))

            ps_1 = ps_1/np.nanmax(ps_1)
            ps_2 = ps_2/np.nanmax(ps_2)
            
            new_labels.append(labels[n])
            new_mags.append(mags[n])
            temp_stack = np.stack((ps_1, ps_2), axis=-1)
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
ps_stack_master_new = np.empty((0, 32, 32, 2))
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
np.save('/Users/mcurrie/GitRepos/TransiNet/data/training_data.npy',
        ps_stack_master_new)
np.save('/Users/mcurrie/GitRepos/TransiNet/data/training_labels.npy',
        labels_master_new)
np.save('/Users/mcurrie/GitRepos/TransiNet/data/training_info.npy',
        info_master_new)
np.save('/Users/mcurrie/GitRepos/TransiNet/data/training_mags.npy',
        mags_master_new)
