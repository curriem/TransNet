import sys
import commands
import glob
import numpy as np
from itertools import combinations, permutations

badpix = sys.argv[1]
set_num = sys.argv[2]
if badpix == 'new':
    badpix_dir = '/sets_newbadpix/'
elif badpix == 'old':
    badpix_dir = '/sets_oldbadpix/'
else:
    assert False

ims = glob.glob('/Volumes/My_Book/TransiNet/data/%s/set_%s_epochs/F*drz*' %
                (badpix_dir, set_num))
print ims
filts = []
epochs = []

for im in ims:
    im = im.split('/')[-1]
    im = im.split('_')
    filt = im[0]
    epoch = im[1]
    epoch = int(epoch.strip('epoch'))
    filts.append(filt)
    epochs.append(epoch)


unique_filts = np.unique(filts)
unique_epochs = np.unique(epochs)

filter_combos = list(combinations(unique_filts, 2))
if len(unique_filts) < 2:
    for filt in unique_filts:
        filter_combos.append((filt, filt))

epoch_combos = list(permutations(unique_epochs, 2))

for filter_combo in filter_combos:
    filt1, filt2 = filter_combo
    for epoch_combo in epoch_combos:
        e1, e2 = epoch_combo
        print 'python check_for_candidates.py %s %s %s %s %s %s' \
                 % (badpix, set_num, filt1,filt2, str(e1),str(e2))

        print commands.getoutput('python check_for_candidates.py %s %s %s \
                                  %s %s %s' % (badpix, set_num,
                                               filt1, filt2,
                                               str(e1), str(e2)))
