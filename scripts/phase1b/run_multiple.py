import sys
import commands
import numpy as np


badpix = sys.argv[1]
script = sys.argv[2]
start = int(sys.argv[3])
stop = int(sys.argv[4])


for n in np.arange(start, stop, 1, dtype=np.int):
    print 'python %s.py %s %s' % (script, badpix, str(n).zfill(3))
    commands.getoutput('python %s.py %s %s' % (script, badpix, str(n).zfill(3)))
