import commands
import sys
from numpy import arange

start = int(sys.argv[1])
stop = int(sys.argv[2])

for n in arange(start, stop, 1, dtype=int):
    commands.getoutput('python transient_search_prep.py %s' % str(n).zfill(3))



