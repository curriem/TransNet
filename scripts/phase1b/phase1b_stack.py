import glob
import numpy as np
import astropy.io.fits as pyfits
import commands
import sys
from drizzlepac import tweakreg, astrodrizzle
import collections
import copy


set_num = sys.argv[1]

def do_it(cmd):
    print cmd
    print commands.getoutput(cmd)

def get_filter(the_header):
    try:
        return the_header["FILTER"]
    except:
        filt = the_header["FILTER1"]
        if filt.find("CLEAR") == -1:
            return filt
        else:
            return the_header["FILTER2"]

def find_filter(flt_list, the_filters):
    if type(the_filters) == type("a"):
        filt_list = [copy.deepcopy(the_filters)]
    else:
        filt_list = copy.deepcopy(the_filters)

    f125w_list = []

    for item in flt_list:
        f = pyfits.open(item)
        if filt_list.count(get_filter(f[0].header)):
            f125w_list.append(item)
    return f125w_list

def find_best_ref(all_flc_list, filt_priority=["F110W", "F105W",
                                               "F140W", "F125W",
                                               "F814W", "F775W",
                                               "F606W", "F160W"]):

    flc_list = []
    for filt in filt_priority:
        if flc_list == []:
            flc_list = find_filter(all_flc_list, filt)
            print "flc_list for ", filt, flc_list

    print "Find ref with least maximum disagreement."
    print "In princple, this should take rotation into account."

    xlist = np.array([], dtype=np.float64)
    ylist = np.array([], dtype=np.float64)

    for fl in flc_list:
        f = pyfits.open(fl)

        ra = f[0].header["RA_TARG"]
        dec = f[0].header["DEC_TARG"]

        f.close()

        x = ra*np.cos(dec/57.3)*3600*20
        y = dec*3600*20

        xlist = np.append(xlist, x)
        ylist = np.append(ylist, y)


    besttotal = 1.e10
    for i in range(len(xlist)):
        new = np.sqrt((xlist - xlist[i])**2. + (ylist - ylist[i])**2.)
        if max(new) < besttotal:
            besttotal = max(new)
            besti = i

    print "Ref to use ", flc_list[besti], besti
    return flc_list[besti], besti

def transfer_header(infl, outfl):
    """I don't know why Eli's version of this doesn't work..."""

    print "Transfer", infl, "to", outfl
    fin = pyfits.open(infl)
    fout = pyfits.open(outfl, 'update')

    dont_transfer = ["HSTSLAC", "MDRIZSKY", "LACOSMIC", "HISTORY", "COMMENT", ""]

    print "Transferring: ",
    for i in range(len(fin)):
        for key in fin[i].header:
            if dont_transfer.count(key) == 0:
                if fin[i].header[key] != fout[i].header.get(key, default = None):
                    print key,

                    fout[i].header[key] = fin[i].header[key]
    fout.flush()
    fout.close()
    fin.close()
    print 

def do_tweak(flt_list, besti, lowthreshold = 0):

    f = open(bad_pix_list_wfc3)
    lines = f.read().split('\n')
    f.close()

    lines = [item.split(None) for item in lines]
    lines = [item for item in lines if item != []]
    bad_pix = [(int(item[0]), int(item[1])) for item in lines]

    tmp_ims = []

    for i in range(len(flt_list)):
        f = pyfits.open(flt_list[i])
        if f[0].header["INSTRUME"] == "ACS":
            tmp_ims.append(flt_list[i].replace(".fits", "_lac.fits"))
            acs = True
        else:
            tmp_ims.append(flt_list[i].replace(".fits", "_filter.fits"))

            if flt_list[i] == tmp_ims[i]:
                print "Error with ", flt_list[i]
                sys.exit(1)

            print "Median Filtering ", flt_list[i]
            f = pyfits.open(flt_list[i])
            tmpdata = copy.deepcopy(f["SCI"].data)

            LTV1 = f["SCI"].header["LTV1"]
            LTV2 = f["SCI"].header["LTV2"]

            for this_x, this_y in bad_pix:

                this_x += LTV1
                this_y += LTV2

                if this_x > 1 and this_x < len(tmpdata[0]) and this_y > 1 and this_y < len(tmpdata):
                    f["SCI"].data[int(np.around(this_y - 1)), int(np.around(this_x - 1))] = np.median(tmpdata[int(np.around(this_y - 2)): int(np.around(this_y + 1)), int(np.around(this_x - 2)): int(np.around(this_x + 1))])
            f.writeto(tmp_ims[i], clobber = True)
            f.close()
            acs = False
        do_it("cp -f " + tmp_ims[i] + " " + tmp_ims[i].replace("/orig_files/", "/"))
        tmp_ims[i] = tmp_ims[i].replace("/orig_files/", "/")


    print "tmp_ims ", tmp_ims

    tweakref = tmp_ims[besti]

    tweakreg.TweakReg(','.join(tmp_ims),
                      updatehdr=True,
                      shiftfile=True, # This is just for show
                      ############ Change This Between Iterations: ##########
                      refimage=tweakref,
                      updatewcs=False, # I think this should always be false.
                      searchrad=4,
                      searchunits='arcseconds',
                      threshold=(1. + 7.*acs)/(lowthreshold + 1.),
                      conv_width=(2.5 + 1.*acs), # 3.5 for optical, 2.5 for IR
                      ######### Change This Between Iterations: ##############
                      wcsname="TWEAK_rough",
                      residplot='No plot',
                      see2dplot=False,
                      fitgeometry='shift') # Have to change this for that one epoch, G cluster?


    f = open("shifts.txt")
    lines = f.read()
    f.close()

    if lines.find(" nan ") != -1:
        print "Couldn't match!"

        if lowthreshold == 0: # First iteration
            print "Trying lower threshold..."
            do_tweak(flt_list, besti, lowthreshold = 1)
        else:
            print "...even though lowthreshold is ", lowthreshold
            sys.exit(1)


    for i in range(len(flt_list)):
        print "Transferring from ", tmp_ims[i], flt_list[i]
        transfer_header(tmp_ims[i], flt_list[i])



def do_drizzle(flc_list, outputname, clean = True, refimage = "", build = True, cr_sensitive = False, outputscale = 0.05):
    print "overriding cr_sensitive", cr_sensitive
    cr_sensitive = True
    n_img = len(flc_list)

    combine_type = "minmed"*(n_img <= 4.) + "median"*(n_img > 4)

    print "Number of images ", n_img, combine_type
    if refimage != "":
        print "Using refimage", refimage

    nicmos = (flc_list[0].split("/")[-1][0] == "n")

    if nicmos:
        combine_type = "minmed"

    wfc3 = (flc_list[0].split("/")[-1][0] == "i")

    print "flc_list, nicmos, wfc3 ", flc_list, nicmos, wfc3
    
    astrodrizzle.AstroDrizzle(','.join(flc_list),
                              preserve=False,
                              build=build,
                              output=outputname,
                              clean=clean*0, # Clean up tmp files
                              updatewcs=nicmos, # This is right
                              proc_unit='native',

                              driz_sep_kernel='square',
                              driz_sep_pixfrac=1.0,
                              driz_sep_scale=0.128,
                              driz_sep_bits=(0 + (512+1024+2048)*nicmos 
                                             + (2048+8192)*wfc3),

                              combine_type=combine_type,
                              driz_cr=(n_img > 1), 
                              median=(n_img > 1),
                              blot=(n_img > 1),
                              static=(n_img > 1), 
                              #driz_cr_snr = "3.5 3.0",
                              driz_cr_scale=("3  2"*(1 - cr_sensitive)
                                             + "2  1.5"*cr_sensitive), # Up from default 1.2, 0.7
                              #driz_cr_scale = "2. 1.5",

                              #final_wht_type = "ERR", # This is very wrong! Why do they even include it?
                              final_wht_type="EXP", # This one works!
                              final_kernel="gaussian",
                              final_pixfrac=1.0, # Should be default.
                              final_wcs=True,
                              final_rot=0.,
                              final_bits=(0 + (512+1024+2048)*nicmos
                                          + (2048+8192)*wfc3),
                              final_scale=outputscale,
                              final_refimage=refimage)

    if nicmos:
        f = pyfits.open(outputname + "_drz.fits", 'update')
        expend = f[0].header["EXPEND"]
        print outputname, "EXPEND", expend
        if expend > 51544:
            print "Multiplying by 1.007!"
            f["SCI"].data *= 1.007
        f.flush()
        f.close()



def get_fls_by_filter_date(globpath = ""):
    files_by_filter_date = collections.OrderedDict()

    if globpath == "":
        origfls = glob.glob(data_path
                            + "set_%s/orig_files/*flt.fits" % set_num)
        simfls = [] #glob.glob("simulated_ims/*flt.fits")
    else:
        origfls = glob.glob(globpath)
        simfls = []

    for i in range(len(origfls))[::-1]:
        foundsim = 0
        for simfl in simfls:
            if origfls[i].split("/")[-1] == simfl.split("/")[-1]:
                foundsim = 1
        if foundsim:
            del origfls[i]

    

    fls_sorted_by_date = []
    for fl in origfls + simfls:
        f = pyfits.open(fl)
        EXPEND = f[0].header["EXPEND"]
        f.close()
        
        fls_sorted_by_date.append((EXPEND, fl))
    
    fls_sorted_by_date.sort()

    # print fls_sorted_by_date
    fls_sorted_by_date = [item[1] for item in fls_sorted_by_date]
    

    for fl in fls_sorted_by_date:
        f = pyfits.open(fl)
        EXPEND = f[0].header["EXPEND"]
        FILTER = f[0].header["FILTER"]
        f.close()

        found = 0
        for key in files_by_filter_date:
            if (key[0] == FILTER) and (abs(EXPEND - key[1]) < 1.):
                files_by_filter_date[key].append(fl)
                found += 1
        assert found < 2
        if found == 0:
            files_by_filter_date[(FILTER, EXPEND)] = [fl]
    # for key in files_by_filter_date:
    #    print key, files_by_filter_date[key]
    return files_by_filter_date

def sort_ims(ims_path):

    origfls = glob.glob(ims_path+'/*flt.fits')
    print origfls
    ims_dict = {}
    for fl in origfls:
        f = pyfits.open(fl)
        EXPEND = int(f[0].header["EXPEND"])
        FILTER = f[0].header["FILTER"]
        f.close()

        just_fl = fl.split('/')[-1]
        print just_fl, FILTER, EXPEND
        try:
            ims_dict[FILTER]
        except:
            ims_dict[FILTER] = {}

        try:
            ims_dict[FILTER][EXPEND].append(just_fl)
        except:
            ims_dict[FILTER][EXPEND] = []
            ims_dict[FILTER][EXPEND].append(just_fl)

    filt1, filt2 = ims_dict.keys()

    filt1_e1 = np.min(ims_dict[filt1].keys())
    filt1_e2 = np.max(ims_dict[filt1].keys())
    filt2_e1 = np.min(ims_dict[filt2].keys())
    filt2_e2 = np.max(ims_dict[filt2].keys())

    filt1_epoch1_fls = ims_dict[filt1][filt1_e1]
    filt1_epoch2_fls = ims_dict[filt1][filt1_e2]
    filt2_epoch1_fls = ims_dict[filt2][filt2_e1]
    filt2_epoch2_fls = ims_dict[filt2][filt2_e2]

    return filt1, filt2, filt1_epoch1_fls, filt1_epoch2_fls, \
           filt2_epoch1_fls, filt2_epoch2_fls


def get_filters(ims_path):
    origfls = glob.glob(ims_path+'/*flt.fits')
    print ims_path
    print origfls
    filts = []
    for fl in origfls:
        f = pyfits.open(fl)
        FILTER = f[0].header["FILTER"]
        f.close()
        filts.append(FILTER)

    unique_filters = np.unique(filts)
    return unique_filters


#path = '/Users/mcurrie/Projects/TransiNet/data/set_%s/orig_files' % set_num

#data_path = '/Users/mcurrie/Projects/TransiNet/data/'
#path = '/Volumes/My_book/TransiNet/data/set_%s/orig_files' % set_num
#data_path = '/Volumes/My_book/TransiNet/data/'
data_path = '/Volumes/My_Book/TransiNet/data/sets_newbadpix/'
# step 0: stack images


with open('obj_coords.dat', 'wb') as f:
    f.write('set_%s 0 0' % set_num)
    
    
    
    
outputscale = 0.09
sky_nlc_order = 'nlcsky'
bad_pix_list_wfc3 = data_path + 'bad_pix_list_wfc3.txt'
set_num = sys.argv[1]
set_dir = 'set_' + set_num
userrefimage = ''

do_it("mkdir %s/%s/orig_files" % (data_path, set_dir))

do_it("mv %s/*fits %s/orig_files" % (data_path + set_dir,
                                     data_path + set_dir))
print "Aligning Images..."
for filter in [["F606W", "F775W", "F814W"],
               ["F105W", "F110W", "F125W", "F140W", "F160W"]]:
    flt_list = glob.glob(data_path + set_dir + "/orig_files/i*flt.fits") + \
            glob.glob(data_path + set_dir + "/orig_files/j*flc.fits")
    flt_list.sort()
    flt_list = find_filter(flt_list, filter)
    if flt_list != []:
        best_ref, besti = find_best_ref(flt_list)
        do_tweak(flt_list, besti)
        do_it("rm -f %s/*.coo %s/*.match %s/*catfile.list" 
              % (data_path + set_dir,
                 data_path + set_dir,
                 data_path + set_dir))
        do_it("mv shifts.txt "
              + data_path + set_dir
              + "/shifts_%s.txt" % "_".join(filter))

print 'Finished alignment'

print "Drizzling WFC3..."

for filter in ["F105W", "F110W", "F125W", "F140W", "F160W"]:
    files = find_filter(glob.glob(data_path + set_dir
                                  + "/orig_files/i*flt.fits"), filter)
    print "filter, files", filter, files

    if len(files) > 0:
        for cr_sensitive in [0]:

            new_files = [item.replace("/orig_files", "") for item in files]

            for file, new_file in zip(files, new_files):
                if new_file == file:
                    print "Error,", new_file, "is the same!"
                    sys.exit(1)
                do_it("cp -vf " + file + " " + new_file)


            driz_filename = filter + "_stack" + "_CRsens"*cr_sensitive
            do_drizzle(new_files,
                       driz_filename,
                       clean=True,
                       refimage=(userrefimage != "None")*userrefimage,
                       build = True,
                       cr_sensitive=cr_sensitive,
                       outputscale=outputscale)
            
            do_it("mv " + driz_filename + "_drz.fits "
                  + data_path + set_dir)
            do_it("rm -fv " + " ".join(new_files))

print "Drizzling ACS..."
for filter in ["F775W", "F814W", "F606W", "F850LP"]:
    files = find_filter(glob.glob(data_path + set_dir
                                  + "/orig_files/j*flc.fits"), filter)
    print "filter, files", filter, files

    if len(files) > 0:
        for cr_sensitive in [0]:

            new_files = [item.replace("/orig_files", "") for item in files]

            for file, new_file in zip(files, new_files):
                if new_file == file:
                    print "Error,", new_file, "is the same!"
                    sys.exit(1)
                do_it("cp -vf " + file + " " + new_file)

            driz_filename = filter + "_stack" + "_CRsens"*cr_sensitive
            do_drizzle(new_files,
                       driz_filename,
                       clean=True,
                       refimage=(userrefimage != "None")*userrefimage,
                       build=True,
                       cr_sensitive=cr_sensitive,
                       outputscale=outputscale)
            do_it("mv " + driz_filename + "_drc.fits "
                  + data_path + set_dir)
            do_it("rm -fv " + " ".join(new_files))

unique_filters = get_filters(data_path+set_dir+'/orig_files/')

origfls = glob.glob(data_path+'/orig_files/*flt.fits')

with open(data_path + 'paramfile_%s.txt' % set_num, 'wb') as paramfl:

    paramfl.write('drz\t%s/set_%s/%s_stack_drz.fits\n' % (data_path,
                                                          set_num,
                                                          unique_filters[0]))
    paramfl.write('aligned\t%s\n' % ' '.join(origfls))
    paramfl.write('F125W_zp\t26.23\n')
    paramfl.write('F105W_zp\t26.24\n')
    paramfl.write('F140W_zp\t26.44\n')
    paramfl.write('F160W_zp\t25.92\n')
    paramfl.write('min_mag\t25.0\n')
    paramfl.write('max_mag\t27.0\n')
    paramfl.write('step_mag\t0.2\n')
    paramfl.write('gauss_r\t4\n')
    paramfl.write('frac_real\t0.5\n')
    paramfl.write('F125W_highz\t26.8\n')
    paramfl.write('F105W_highz\t26.8\n')
    paramfl.write('F140W_highz\t26.0\n')
    paramfl.write('F160W_highz\t25.9\n')
    paramfl.write('frac_highz\t0.003\n')

# stack epochs

fls_by_filter_date = get_fls_by_filter_date()

commands.getoutput("rm -f %s/set_%s_epochs/*" % (data_path, set_num))
commands.getoutput("mkdir %s/set_%s_epochs" % (data_path, set_num))

filter_counter = []

for item in fls_by_filter_date:
    print item

    for im in fls_by_filter_date[item]:
        commands.getoutput("cp " + im + " %s/set_%s_epochs" % (data_path,
                                                               set_num))

    filter_counter.append(item[0])

    refimage = commands.getoutput("grep drz "+data_path+"paramfile_%s.txt" %
                                  set_num).split(None)[1] + "[SCI]"
    print "refimage", refimage

    do_drizzle([data_path + "set_"+set_num+"_epochs/" + subitem.split("/")[-1] for subitem in fls_by_filter_date[item]],
               outputname = data_path + "set_"+set_num+"_epochs/" + item[0] + "_epoch%02i" % (filter_counter.count(filter_counter[-1])),
               refimage=refimage,
               outputscale=outputscale)


