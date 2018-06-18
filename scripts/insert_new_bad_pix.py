import glob
import astropy.io.fits as pyfits
import numpy as np

dat = np.loadtxt('../../wfc3_bad_pix/data/2017_bad_pix_list.txt')
dat = dat.T.astype(int)
x, y = dat
fits_x = y + 1
fits_y = x + 1

#dirs = glob.glob('/Volumes/My_Book/TransiNet/data/orig_newbadpix_sets/set*')
dirs = glob.glob('../../step_1/set_*/')
dirs = glob.glob('/Volumes/My_Book/TransiNet/data/set_*')
for dir in dirs:
    print dir
    ims = glob.glob(dir + '/*')
    for im in ims:
        print im
        with pyfits.open(im, mode='update') as fits_im:
            fits_im[3].data[x, y] = 4
            fits_im.flush()
