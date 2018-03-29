import numpy as np
import transinet_funcs as transf
import astropy.io.fits as pyfits
from astropy import wcs
import sys
import keras
import matplotlib.pyplot as plt
import sep
import commands

'''this subtracts two ims, finds objects with sep and checks the objects as
it finds them'''



def plot_candidate(im1a_path, im1b_path, im2a_path, im2b_path,
                   pix_x, pix_y, score, save_path, n):
    im1a = pyfits.open(im1a_path)
    im1b = pyfits.open(im1b_path)
    im2a = pyfits.open(im2a_path)
    im2b = pyfits.open(im2b_path)
    w = wcs.WCS(im1a[1].header)
    ra, dec = w.all_pix2world(pix_x, pix_y, 0)
    im1a_data = im1a[1].data
    im1b_data = im1b[1].data
    im2a_data = im2a[1].data
    im2b_data = im2b[1].data
    e2_date = im1b[0].header['EXPEND']
    e1_date = im1a[0].header['EXPEND']

    im1a_stamp = transf.make_stamp(pix_x, pix_y, im1a_data)
    im1b_stamp = transf.make_stamp(pix_x, pix_y, im1b_data)
    im2a_stamp = transf.make_stamp(pix_x, pix_y, im2a_data)
    im2b_stamp = transf.make_stamp(pix_x, pix_y, im2b_data)
    
    sub1 = im1a_stamp - im1b_stamp
    sub2 = im2a_stamp - im2b_stamp

    fig, ax = plt.subplots(3, 2, figsize=(4,6),
                           sharey=True)
    ax[0, 0].imshow(im1a_stamp, cmap='gray')
    ax[0, 1].imshow(im2a_stamp, cmap='gray')
    ax[1, 0].imshow(im1b_stamp, cmap='gray')
    ax[1, 1].imshow(im2b_stamp, cmap='gray')
    ax[2, 0].imshow(sub1, cmap='gray')
    ax[2, 1].imshow(sub2, cmap='gray')
    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    ax[2, 0].axis('off')
    ax[2, 1].axis('off')
    ax[0, 0].text(-8, 13, 'Epoch 1\nmjd: %s' % int(e1_date),
                  rotation=90)
    ax[1, 0].text(-8, 13, 'Epoch 2\nmjd: %s' % int(e2_date),
                  rotation=90)
    ax[2, 0].text(-8, 13, 'subtraction\nE1-E2',
                  rotation=90)
    ax[2, 0].text(0, 40, 'score: %.03f\nRA: %.03f\nDec: %.03f' % (score[0], ra,
                                                                 dec))
    ax[0, 0].set_title(filt1)
    ax[0, 1].set_title(filt2)
    plt.savefig(save_path+str(n)+'.pdf')


def run_sep(sub):

    pix_xs = []
    pix_ys = []
    extract = sep.extract(sub, thresh=0.035)
    for obj in extract:
        x = int(obj[7])
        y = int(obj[8])
        if np.isnan(np.sum(sub[y-20:y+21, x-20:x+21])):
            continue
        else:
            pix_xs.append(x)
            pix_ys.append(y)
    coords = np.array([pix_xs, pix_ys])
    coords = coords.T
    return coords


def subtract_ims(im1_path, im2_path):
    im1 = pyfits.open(im1_path)
    im2 = pyfits.open(im2_path)
    data1 = im1[1].data
    data2 = im2[1].data
    im1.close()
    im2.close()
    sub = data1 - data2
    #hdu = pyfits.PrimaryHDU(sub)
    #hdu.writeto('test.fits')
    #commands.getoutput('ds9 test.fits')
    #plt.figure()
    #plt.imshow(sub, cmap='gray')
    #plt.colorbar()
    #plt.show()
    return sub

model_path = '../../model/transinet_v1.h5'
model = keras.models.load_model(model_path)
set_num = sys.argv[1]
filt1 = sys.argv[2]
filt2 = sys.argv[3]
epoch1 = sys.argv[4]
epoch2 = sys.argv[5]

save_path = \
  '/Users/mcurrie/Projects/TransiNet/plots/set_%s/%s_%s_%s-%s_cand' \
            % (set_num, filt1, filt2, epoch1, epoch2 )

master_path = '/Volumes/My_Book/TransiNet/data/'
set_path = master_path + 'set_%s_epochs/' % set_num.zfill(3)
im1a_path = set_path + '%s_epoch0%s_drz.fits' % (filt1, epoch1)
im1b_path = set_path + '%s_epoch0%s_drz.fits' % (filt1, epoch2)
im2a_path = set_path + '%s_epoch0%s_drz.fits' % (filt2, epoch1)
im2b_path = set_path + '%s_epoch0%s_drz.fits' % (filt2, epoch2)
print im1a_path
print im1b_path
print im2a_path
print im2b_path

sub1 = subtract_ims(im1a_path, im1b_path)
sub2 = subtract_ims(im2a_path, im2b_path)
stamp_size=14
coords1 = run_sep(sub1)
coords2 = run_sep(sub2)
coords = np.concatenate((coords1, coords2))

if coords.shape[0] == 0:
    print 'No objects found!'
n = 0
for coord in coords:
    x, y = coord
    try:
        stamp1 = transf.make_stamp(int(x), int(y), sub1, stamp_size)
        stamp2 = transf.make_stamp(int(x), int(y), sub2, stamp_size)
        stamp1_temp = stamp1 + np.abs(np.nanmin(stamp1))
        stamp2_temp = stamp2 + np.abs(np.nanmin(stamp2))
        stamp1_temp /= np.nanmax(stamp1_temp)
        stamp2_temp /= np.nanmax(stamp2_temp)
        obj_stamps = np.stack((stamp1_temp, stamp2_temp), axis=-1)
        # run through cnn
        probs = model.predict(np.array([obj_stamps]))
        prob_of_SN = probs.T[0]

    except:
        print 'passing'
        pass
    if prob_of_SN > 0.97:
        print 'found a candidate'
        plot_candidate(im1a_path, im1b_path, im2a_path, im2b_path,
                       x, y, prob_of_SN, save_path, n)
        n+=1
