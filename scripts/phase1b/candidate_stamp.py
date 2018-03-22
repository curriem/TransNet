import numpy as np
import astropy.io.fits as pyfits
import sys
import transinet_funcs as transf
import matplotlib.pyplot as plt

set_num = sys.argv[1]
x_coord = int(sys.argv[2])
y_coord = int(sys.argv[3])
filt1 = sys.argv[4]
filt2 = sys.argv[5]

fl_path = '/Users/mcurrie/Projects/TransiNet/data/set_%s_epochs/' % set_num

f1_e1 = pyfits.open(fl_path + filt1 + '_epoch01_drz.fits')
f1_e2 = pyfits.open(fl_path + filt1 + '_epoch02_drz.fits')
f2_e1 = pyfits.open(fl_path + filt2 + '_epoch01_drz.fits')
f2_e2 = pyfits.open(fl_path + filt2 + '_epoch02_drz.fits')

f1_e1_data = f1_e1[1].data
f1_e2_data = f1_e2[1].data
f2_e1_data = f2_e1[1].data
f2_e2_data = f2_e2[1].data

e2_date = f1_e2[0].header['EXPEND']
e1_date = f1_e1[0].header['EXPEND']

f1_e1_stamp = transf.make_stamp(x_coord, y_coord, f1_e1_data)
f1_e2_stamp = transf.make_stamp(x_coord, y_coord, f1_e2_data)
f2_e1_stamp = transf.make_stamp(x_coord, y_coord, f2_e1_data)
f2_e2_stamp = transf.make_stamp(x_coord, y_coord, f2_e2_data)
sub_f1 = f1_e1_stamp - f1_e2_stamp
sub_f2 = f2_e1_stamp - f2_e2_stamp


fig, ax = plt.subplots(3, 2, figsize=(4,6),
                      sharey=True)
ax[0, 0].imshow(f1_e1_stamp, cmap='gray')
ax[0, 1].imshow(f2_e1_stamp, cmap='gray')
ax[1, 0].imshow(f1_e2_stamp, cmap='gray')
ax[1, 1].imshow(f2_e2_stamp, cmap='gray')
ax[2, 0].imshow(sub_f1, cmap='gray')
ax[2, 1].imshow(sub_f2, cmap='gray')
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
ax[2, 0].text(-8, 13, 'subtraction\nE1 - E2',
              rotation=90)

ax[0, 0].set_title(filt1)
ax[0, 1].set_title(filt2)
plt.savefig('../../plots/candidate_stamps_set_004.pdf')
plt.show()

