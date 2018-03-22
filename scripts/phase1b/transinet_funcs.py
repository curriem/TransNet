import astropy.io.fits as pyfits
import glob
import collections


def load_fits(fl_path):
    data = pyfits.open(fl_path)
    data = data[1].data
    return data


def make_stamp(x, y, im, size=15):

    lower_x = x - size
    upper_x = x + size
    lower_y = y - size
    upper_y = y + size

    stamp = im[lower_y:upper_y, lower_x:upper_x]
    return stamp

def get_fls_by_filter_date(fls_path):
    files_by_filter_date = collections.OrderedDict()

    orig_fls = glob.glob(fls_path)
    print orig_fls
    fls_sorted_by_date = []
    for fl in orig_fls:
        f = pyfits.open(fl)
        EXPEND = f[0].header["EXPEND"]
        FILTER = f[0].header["FILTER"]
        f.close()
        fls_sorted_by_date.append((EXPEND, fl))

    fls_sorted_by_date.sort()
    fls_sorted_by_date = [item[1] for item in fls_sorted_by_date]
    for fl in fls_sorted_by_date:
        f = pyfits.open(fl)
        EXPEND = f[0].header["EXPEND"]
        FILTER = f[0].header["FILTER"]
        f.close()

        found = 0
        for key in files_by_filter_date:
            if (key[0] == FILTER) and (abs(EXPEND - key[1]) < 10.):
                files_by_filter_date[key].append(fl)
                found += 1
        assert found < 2
        if found == 0:
            files_by_filter_date[(FILTER, EXPEND)] = [fl]
    for key in files_by_filter_date.keys():
        print key
    return files_by_filter_date




if __name__ == '__main__':
    get_fls_by_filter_date('/Users/mcurrie/GitRepos/step_1/set_play3/orig_files/*flt.fits')


