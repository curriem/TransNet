import numpy as np
import matplotlib.pyplot as plt
import pickle

path = '/Users/mcurrie/Desktop/wfc3_all_ims.txt'
counter = 0

ims = dict()

ims['dataset'] = []
ims['target_name'] = []
ims['RA'] = []
ims['Dec'] = []
ims['stop_date'] = []
ims['stop_time'] = []
ims['exp_time'] = []
ims['filter'] = []


with open(path, 'rb') as fl:
    for line in fl:
        line = line.split(None)
        if len(line) == 8:
            counter += 1
            dataset_temp = line[0]
            target_name_temp = line[1]
            RA_temp = np.float(line[2])
            Dec_temp = np.float(line[3])
            stop_date_temp = line[4]
            stop_time_temp = line[5]
            exp_time_temp = line[6]
            filter_temp = line[7]

            ims['dataset'].append(dataset_temp)
            ims['target_name'].append(target_name_temp)
            ims['RA'].append(RA_temp)
            ims['Dec'].append(Dec_temp)
            ims['stop_date'].append(stop_date_temp)
            ims['stop_time'].append(stop_time_temp)
            ims['exp_time'].append(exp_time_temp)
            ims['filter'].append(filter_temp)



        else:
            print 'SKIPPING:', line
            pass
for key in ims.keys():
    ims[key] = np.array(ims[key])
pickle.dump(ims, open('../../data/wfc3_data.p', 'wb'))
