import commands

runs = range(10)
scales = range(4)
for run in runs:
    for scale in scales:
        commands.getoutput('python deep_dream.py subdat.fits'
                           + ' ../model/sets_newbadpix/transinet_v2.h5 %i %i'
                           % (scale, run))
