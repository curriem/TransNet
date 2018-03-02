#import stack_epochs
import glob
import numpy as np
import astropy.io.fits as pyfits
import commands
import sys

set_num = sys.argv[1]
def sort_ims(ims_path):

    origfls = glob.glob(ims_path+'/*flt.fits')
    print origfls
    ims_dict = {}
    for fl in origfls:
        f = pyfits.open(fl)
        EXPEND = int(f[0].header["EXPEND"])
        FILTER = f[0].header["FILTER"]
        f.close()

        just_fl = fl.split('/')[-1]
        print just_fl, FILTER, EXPEND
        try:
            ims_dict[FILTER]
        except:
            ims_dict[FILTER] = {}

        try:
            ims_dict[FILTER][EXPEND].append(just_fl)
        except:
            ims_dict[FILTER][EXPEND] = []
            ims_dict[FILTER][EXPEND].append(just_fl)

    filt1, filt2 = ims_dict.keys()

    filt1_e1 = np.min(ims_dict[filt1].keys())
    filt1_e2 = np.max(ims_dict[filt1].keys())
    filt2_e1 = np.min(ims_dict[filt2].keys())
    filt2_e2 = np.max(ims_dict[filt2].keys())

    filt1_epoch1_fls = ims_dict[filt1][filt1_e1]
    filt1_epoch2_fls = ims_dict[filt1][filt1_e2]
    filt2_epoch1_fls = ims_dict[filt2][filt2_e1]
    filt2_epoch2_fls = ims_dict[filt2][filt2_e2]

    return filt1, filt2, filt1_epoch1_fls, filt1_epoch2_fls, \
           filt2_epoch1_fls, filt2_epoch2_fls


path = '/Users/mcurrie/GitRepos/step_1/set_%s/orig_files' % set_num

#fls_by_filter_date = stack_epochs.get_fls_by_filter_date(path)

'''
counter = 0
for item in fls_by_filter_date:
    print counter
    print item, fls_by_filter_date[item]
    counter += 1

keys = fls_by_filter_date.keys()
'''
# step 0: stack images


with open('obj_coords.dat', 'wb') as f:
    f.write('set_%s 0 0' % set_num)

print commands.getoutput('python ../nic2-calibration/STEP2_process_ims.py')


filt1, filt2, filt1_epoch1_fls, filt1_epoch2_fls, filt2_epoch1_fls,\
filt2_epoch2_fls = sort_ims(path)

filters = [filt1, filt2]
e2_fls = filt1_epoch2_fls + filt2_epoch2_fls
e2_fls_new = []
for item in e2_fls:
    e2_fls_new.append('set_%s/orig_files/' % set_num +item)
e2_fls = e2_fls_new

with open('paramfile.txt', 'wb') as paramfl:

    paramfl.write('drz\tset_%s/%s_stack_drz.fits\n' % (set_num, filters[1]))
    paramfl.write('aligned\t%s\n' % ' '.join(e2_fls))
    paramfl.write('F125W_zp\t26.23\n')
    paramfl.write('F105W_zp\t26.24\n')
    paramfl.write('F140W_zp\t26.44\n')
    paramfl.write('F160W_zp\t25.92\n')
    paramfl.write('min_mag\t25.0\n')
    paramfl.write('max_mag\t27.0\n')
    paramfl.write('step_mag\t0.2\n')
    paramfl.write('gauss_r\t4\n')
    paramfl.write('frac_real\t0.5\n')
    paramfl.write('F125W_highz\t26.8\n')
    paramfl.write('F105W_highz\t26.8\n')
    paramfl.write('F140W_highz\t26.0\n')
    paramfl.write('F160W_highz\t25.9\n')
    paramfl.write('frac_highz\t0.003\n')
# step 1: simulate SNe
print commands.getoutput('python ../nic2-calibration/sim_sne_step1.py paramfile.txt')
# step 2: inject SNe into ims

print commands.getoutput('python ../nic2-calibration/sim_sne_step2.py \
                    simulated_ims/simulated.cat %s' % set_num)

print commands.getoutput('python ../nic2-calibration/stack_epochs.py %s' %
                         set_num)
commands.getoutput('rm i* *fits ')
