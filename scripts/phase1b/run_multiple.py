import sys
import commands
import numpy as np


script = sys.argv[1]
start = int(sys.argv[2])
stop = int(sys.argv[3])

for n in np.arange(start, stop, 1, dtype=np.int):
    print 'python %s.py %s' % (script, str(n).zfill(3))
    commands.getoutput('python %s.py %s' % (script, str(n).zfill(3)))
