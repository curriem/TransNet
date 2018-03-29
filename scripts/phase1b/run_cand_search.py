import sys
import commands
import glob
import numpy as np
set_num = sys.argv[1]

ims = glob.glob('/Volumes/My_Book/TransiNet/data/set_%s_epochs/F*drz*' %
                set_num)

filts = []
epochs = []

for im in ims:
    im = im.split('/')[-1]
    im = im.split('_')
    filt = im[0]
    epoch = im[1]
    epoch = int(epoch.strip('epoch'))
    print filt, epoch
    filts.append(filt)
    epochs.append(epoch)

unique_filts = np.unique(filts)
unique_epochs = np.unique(epochs)

for filt1 in unique_filts:
    for filt2 in unique_filts:
        for e1 in unique_epochs:
            for e2 in unique_epochs:
                commands.getoutput('python check_for_candidates.py %s %s \
                                    %s %s %s' % (set_num, filt1, filt2, e1, e2))
