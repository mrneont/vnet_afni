#!/usr/bin/env python

import sys

# on most systems, this should come from regular AFNI install
sys.path.append('/Users/narayanaswamyy2/abin')
from afnipy import afni_base as ab

import os
import argparse  as argp
import json
import random
import numpy     as np
import glob
import re                    # regular expression
import textwrap              # for help text formatting

# ============================================================================
# program version and brief notes on updates

version = '1.1' # adding in more help files
version = '1.2' # match/case -> if/elif

# ============================================================================
# default options and definitions

# default values for program opts
DEF = {
        'verb' : 1,
        'num_cp' : 3,
        'seed_num' : 42,
        'exec_mode' : 'None',
        'input_dir' : '',
        'output_dir' : '',
}

# list of allowed execution modes
LIST_exec_mode = ['None', 'swarm', 'shell']
STR_exec_mode  = ', '.join(LIST_exec_mode)

# default swarm script
scr_swarm = 'master_script.tcsh'
run_swarm = 'run_swarm.tcsh'
cmd_swarm = """#!/bin/tcsh

swarm                                                              \\
    -f  {scr_swarm}                                                \\
    --module afni                                                  \\
    --partition=norm,quick                                         \\
    --threads-per-process=4                                        \\
    --gb-per-process=3                                             \\
    --time=00:30:00                                                \\
    --logdir={cdir_log}                                            \\
    --job-name=job_{cmd}                                           \\
    --merge-output                                                 \\
    --usecsh
""".format(cmd='vnet_aug', scr_swarm=scr_swarm, cdir_log='../logs')

# ============================================================================
# help text and items

dent = '\n' + 5*' '

help_dict = {
    'ddashline' : '='*76,
    'ver'       : version,
}

help_str_top = '''
Overview ~1~

This is a wrapper program which nests the modular data augmentation
scripts.

The current list of data augmentation types supported are:
1) gibbs artifact
2) affine_transformations 
3) gain_inhomogenity
4) zipper noise
5) various % of noise added to dataset
6) refacing 

{ddashline}

Options ~1~

'''.format(**help_dict)

help_str_epi = '''
{ddashline}

Notes on usage ~1~

*****

{ddashline}

Examples ~1~

*****

Functionality to be added ~1~

This option:
 -smallrange = Set all the parameter ranges to be smaller (about half) than
               the default ranges, which are rather large for many purposes.
               * Default angle range    is plus/minus 30 degrees
               * Default shift range    is plus/minus 32% of grid size
               * Default scaling range  is plus/minus 20% of grid size
               * Default shearing range is plus/minus 0.1111

{ddashline}

written by  : Y Swamy (SSCC, NIMH, NIH, USA)
              RC Reynolds (SSCC, NIMH, NIH, USA)
              PA Taylor (SSCC, NIMH, NIH, USA)

current ver :

{ddashline}
'''.format(**help_dict)

# ============================================================================
# ============================================================================

# List of data augmentation in different phases 
list_daug_ph0    = ['reface','empty']
list_daug_ph1    = ['gibbs','affine','empty']
list_daug_ph2    = ['gain_inhom','zipper','add_noise','contrast_var']
list_daug = []

# --------------------------------------------------------------------------

def is_valid_exec_mode(mode):
    """Is the given string 'mode' an allowed execution mode?  Exit on
failure."""

    if not(mode in LIST_exec_mode) :
        print("** ERROR: this is not a valid execution mode: {}"
              "".format(mode))
        sys.exit(7)

    return 0

def get_aug_script_dir():
    """Use built-in Python functions to get the directory where this
script lives, so we can copy augmentation scripts from it."""

    full_path_scr = os.path.abspath(__file__)

    path_dir_list = full_path_scr.split('/')[:-1]

    if not(len(path_dir_list)) :
        print("** ERROR: unable to parse full script path:", full_path_scr)
        sys.exit(4)
    
    # reassemble path to this dir
    path_dir_name = '/'.join(path_dir_list)

    return path_dir_name

def copy_all_aug_script(idir, odir):
    """Copy all augmentation scripts from input dir 'idir' to output dir
'odir'."""

    if not(os.path.exists(idir)):
        print("** ERROR: this input path does not exist:", idir)
        sys.exit(5)
    if not(os.path.exists(odir)):
        print("** ERROR: this output path does not exist:", odir)
        sys.exit(5)

    cmd = """\\cp {idir}/*.tcsh {odir}/.""".format(idir=idir, odir=odir)

    com  = ab.shell_com(cmd, capture=1)
    stat = com.run()
    # print the status of preparing the data_augmentation folder
    if (stat == 0):
        print("++ Copied augmentation scripts to odir:")
        print("   {odir}".format(odir=odir))
    else :
        print("** ERROR: failed to copy augmentation scripts to odir:")
        print("   {odir}".format(odir=odir))
        sys.exit(6)

    return 0

def get_data_augment_args():
    """Set of options to be used on the command line, with defaults and
short help descriptions for each.
    """

    # overall help formatting, and add chunks of text to option list
    parser = argp.ArgumentParser(prog=str(sys.argv[0]).split('/')[-1],
                        add_help=False,
                        formatter_class=argp.RawDescriptionHelpFormatter,
                        description=textwrap.dedent(help_str_top),
                        epilog=textwrap.dedent(help_str_epi) )

    # data_augment_path is the data_path for data augmentation folder
    parser.add_argument("-input_dir", nargs=1,
                        default=[DEF['input_dir']],
                        help='(req) path for input dir, likely a directory '
                        'that ends with "/training", which includes "mask" '
                        'and "orig" subdirectories '
                        '(def: {})'.format(DEF['input_dir']))

    # data_dir is the data path for the dataset folder
    parser.add_argument("-output_dir", nargs=1,
                        default=[DEF['output_dir']],
                        help='(req) path for output dir, likely a directory '
                        'that ends with "/training", which will be populated '
                        'with these subdirectories: "mask", "orig", "scripts", '
                        '"logs", and, if executing, "edt" '
                        '(def: {})'.format(DEF['output_dir']))

    # number of copies of the dataset to be made 
    parser.add_argument("-num_cp", nargs=1,
                        default=[DEF['num_cp']],
                        help='number of total copies of a dset to have int '
                        'the augmented set, including the original '
                        '(def: {})'.format(DEF['num_cp']))

    # random seed number (integer)
    parser.add_argument("-seed_num", nargs=1,
                        default=[DEF['seed_num']],
                        help='seed number for randomization of augmentation '
                        'steps '
                        '(def: {})'.format(DEF['seed_num']))

    # execution mode for augmentation script set
    parser.add_argument("-exec_mode", nargs=1,
                        default=[DEF['exec_mode']],
                        help='execution mode for data augmentation scripts, '
                        'from among: {} '
                        '(def: {})'.format(STR_exec_mode, DEF['exec_mode']))

    # verbosity level (integer)
    parser.add_argument("-verb", nargs=1,
                        default=[DEF['verb']],
                        help='verbosity level '
                        '(def: {})'.format(DEF['verb']))

    parser.add_argument('-help', action="store_true", 
                        default=False,
                        help='display help in terminal') 

    parser.add_argument('-hview', action="store_true", 
                        default=False,
                        help='display help in a text editor') 

    args = parser.parse_args()

    # display program version (later, when an AFNI program, do_view differs)
    do_help  = args.help
    do_hview = args.hview
    if len(sys.argv) == 1 or do_help or do_hview :
        parser.print_help()
        sys.exit(0)

    # required args
    if len(args.input_dir) == 0 :
        print("** ERROR: need to provide an '-input_dir ..'")
        sys.exit(8)
    if len(args.output_dir) == 0 :
        print("** ERROR: need to provide an '-output_dir ..'")
        sys.exit(8)

    return parser.parse_args()


def get_daug_dict(fl_basename): 
    
    global daug1_keys, daug1_values, daug_ph1_val
    global daug2_keys, daug2_values, daug_ph2_val

    daug1_keys   = []
    daug1_values = []
    daug2_keys   = []
    daug2_values = []
    list_param1D = []

    #do phase0 data augmentation 
    # phase0 -> refacing is done randomly to 20% of da 
    phase0_rand_item = random.choices([0,1], weights=(20, 80))
    phase0_rand      = phase0_rand_item[0]
    phase0_daug      = list_daug_ph0[phase0_rand]
    print('++ phase_0: ', phase0_daug)
    daug0_keys       = ['type']
    daug0_values     = [phase0_daug]

    
    # ----- Nested dict for parameters
    
    # do phase1 data augmentation 
    # + phase1 augmentation is randomly picked from 'list_daug_ph1'
    # + the two types of augmentations in phase 1 are gibbs and affine
    # + 15% weightage is given to gibbs and 
    #   75% weightage os given to affine 
    # + phase1_rand_item is either 0 or 1. 
    # + 0 is given 15% weightage and 1 is given 75% weightage 
    phase1_rand_item = random.choices([0,1,2], weights=(15, 45,40))
    phase1_rand      = phase1_rand_item[0]
    phase1_daug      = list_daug_ph1[phase1_rand]
    print('++ phase_1 :', phase1_daug)
    daug1_keys       = ['type']
    daug1_values     = [phase1_daug]
    
    # do phase2 data augmentation 
    # ['gain_inhom','zipper','add_noise','contrast_var']
    phase2_rand_item = random.choices([0,1,2,3], weights=(30, 10, 30, 30))
    phase2_rand      = phase2_rand_item[0]
    phase2_daug      = list_daug_ph2[phase2_rand]
    print('++ phase_2: ', phase2_daug)

    # ----- Phase 1

    if phase1_daug == 'gibbs' :
        print("augmentation type : doing gibbs")
        radius   = random.randint(70, 100)
        daug1_keys = ['type', 'radius']
        daug1_values = [phase1_daug, radius]

    elif phase1_daug == 'affine' :
        # Default shift range    is plus/minus 32% of grid size
        # Default scaling range  is plus/minus 20% of grid size
        # Default shearing range is plus/minus 0.1111
        # Default angle range    is plus/minus 30 degrees
        # yns: option to pick without replacement 
        affine_rand = random.choices(['rotation','scale','shear','shift'], k=2)
        # condition to avoid random selection of same two augmentation types
        while (affine_rand[0] == affine_rand[1]):
            affine_rand = random.choices(['rotation','scale','shear','shift'], k=2)

        affine_type = affine_rand[0]+'_'+ affine_rand[1]
        # DEFINITION OF AFFINE TRANSFORMATION PARAMETERS
        #     x-shift   y-shift   z-shift 
        #     z-angle   x-angle   y-angle 
        #     x-scale   y-scale   z-scale 
        #     y/x-shear z/x-shear z/y-shear 
        # param_map = [#1  #2  #3  #4 #5 #6 #7  #8  #9  #10   #11   #12]
        # param_map = [Shx Shy Shz Rz Rx Ry Scx Scy Scz Sheyx Shezx Shezy]

        # empty list for param1D of 2 types of affine_transforms
        param1D=[[],[]]
        # for loop over the     
        for x in range(len(affine_rand)):
            list_axis = ['x', 'y', 'z']

            if affine_rand[x] == 'rotation' :
                # Default angle range is plus/minus 30 degrees
                rand_axis = random.choice(list_axis)
                rand_rot  = random.randint(-30, 30)
                # condition to avoid the rotation = 0
                while (rand_rot == 0):
                    rand_rot = random.randint(-30, 30)
                if(rand_axis == 'x'):
                    Rx = rand_rot
                    Rz = 0
                    Ry = 0
                elif (rand_axis == 'y'):
                    Ry = rand_rot
                    Rx = 0
                    Rz = 0
                else : # (rand_axis == 'z'):
                    Rz = rand_rot
                    Rx = 0
                    Ry = 0
                print("augmentation type : doing rotation")
                param1D[x] = [0, 0, 0, Rz, Rx, Ry, 0, 0, 0, 0, 0, 0]
                    
            elif affine_rand[x] == 'shift' :
                print("augmentation type : doing shift")
                list_shift = [-5,-10,-15,-20,-25,-30,5,10,15,20,25,30]
                rand_shift = random.choice(list_shift)
                rand_axis  = random.choice(list_axis)
                if(rand_axis == 'x'):
                    Shx = rand_shift
                    Shy = 0
                    Shz = 0
                elif (rand_axis == 'y'):
                    Shx = 0
                    Shy = rand_shift
                    Shz = 0
                else : # (rand_axis == 'z'):
                    Shx = 0
                    Shy = 0
                    Shz = rand_shift
                
                param1D[x] = [Shx, Shy, Shz, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                
            elif affine_rand[x] == 'scale' :
                print("augmentation type : doing scale")
                # scaling applies to all axis;
                # yns range is (0.8,1.2) 
                rand_scale = round(random.uniform(0.8, 1.2), 1)
                Scz        = rand_scale
                Scx        = rand_scale
                Scy        = rand_scale
                
                param1D[x] = [ 0, 0, 0, 0, 0, 0, Scx, Scy, Scz, 0, 0, 0]
                    
            elif affine_rand[x] == 'shear' :
                print("augmentation type : doing shear")
                rand_axis  = random.choice(list_axis)
                rand_shear = round(random.uniform(-0.1, 0.1), 2)
                # condition to avoid the scaling =0
                while (rand_shear == 0):
                    rand_shear = round(random.uniform(-0.1, 0.1), 2)
                if(rand_axis == 'x'):
                    Shex = rand_shear
                    Shey = 0
                    Shez = 0
                elif (rand_axis == 'y'):
                    Shex = 0
                    Shey = rand_shear
                    Shez = 0
                else : # (rand_axis == 'z'):
                    Shex = 0
                    Shey = 0
                    Shez = rand_shear

                param1D[x] = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, Shex, Shez, Shey]
                    
        #adding element wise the param list of the two  affine transformations
        list_param1D = [x + y for x, y in zip(param1D[0], param1D[1])]
        param1D      =  " ".join(map(str, list_param1D))
        daug1_keys   = ['type', 'param1D']
        daug1_values = [phase1_daug, param1D]
        print(param1D)

    # ----- Phase 2 

    if phase2_daug == 'gain_inhom' :
        print("augmentation type : doing gain_inhom")
        # window over which the scaling factor varies:
        # the scaling factor varies between values 'win_min' and 'win_max'
        # within [2,2.8]
        list_axis    = ['i','j','k']
        rand_axis    = random.choices(list_axis,weights=(10, 80, 10))
        win_min      = 2
        win_max      = 2.8
        #Returns a random float number up to 1 decimal places
        window       = round(random.uniform(win_min, win_max), 1)
        daug2_keys   = ['type', 'axis', 'window']
        daug2_values = [phase2_daug, rand_axis, window]

    elif phase2_daug == 'zipper' :
        print("augmentation type : doing zipper")
        list_axis    = ['i','j','k']
        rand_axis    = random.choice(list_axis)
        zip_width    = 5
        daug2_keys   = ['type', 'axis', 'zip_width']
        daug2_values = [phase2_daug, rand_axis, zip_width]

    elif phase2_daug == 'add_noise' :
        # noise level of 10% 20% 30%
        print("augmentation type : doing add_noise")
        noise_lvl      = [0.1, 0.2, 0.3]
        rand_noise_lvl = random.choice(noise_lvl)
        list_axis      = ['i','j','k']
        rand_axis      = random.choice(list_axis)
        daug2_keys     = ['type', 'axis', 'noise_lvl']
        daug2_values   = [phase2_daug, rand_axis, rand_noise_lvl]

    elif phase2_daug == 'contrast_var' :
        print("augmentation type : doing contrast_var")
        shading      = random.randint(0, 1)
        daug2_keys   = ['type', 'shading']
        daug2_values = [phase2_daug, shading]

    # ----- end of Phase 2

    keys = ['fl_basename', 'daug_ph0', 'daug_ph1', 'daug_ph2']
    daug_ph0_val = dict(zip(daug0_keys, daug0_values))
    daug_ph1_val = dict(zip(daug1_keys, daug1_values))
    daug_ph2_val = dict(zip(daug2_keys, daug2_values))
    values = [fl_basename, daug_ph0_val, daug_ph1_val, daug_ph2_val]
    
    daug_dict = dict(zip(keys, values))

    return daug_dict


def write_script(da_dict,fl_name, daug):
    # set the filename for the shell script based on the dataset
    dir_name  = os.path.dirname(fl_name)
    file_name = os.path.basename(fl_name)
    
    new_extension = ".tcsh"
    #print('file_name=',file_name)
    script_fl = file_name.split('.')[0] +new_extension
    script_fl_path_str = os.path.join(os.path.dirname(dir_name), 'scripts', script_fl)
    print('script_fl =', script_fl)
    f = open(script_fl_path_str, "w")
    f.write("#!/bin/tcsh")
    f.write('\n')

    # notes 
    # + the individual shell script need the 
    #   absolute path to the dataset 
    #print("write_script fl_name =  ",fl_name)


    

    # phase-1  data augmentation shell script
    if (daug == 1):

        # phase-0  
        if (da_dict['daug_ph0']['type'] == 'reface'):
            f.write("tcsh do_reface.tcsh {}""".format(fl_name))
            f.write('\n')

        print('\n')
        print('DAUG1 = ',da_dict['daug_ph1']['type'])

        if (da_dict['daug_ph1']['type'] == 'gibbs'):#(weightage =15%)
            gibbs_radius = da_dict['daug_ph1']['radius']
            print('gibbs_radius =',gibbs_radius)
            f.write("tcsh do_gibbs.tcsh {} {}""".format(fl_name,\
                                                        gibbs_radius))
            f.write('\n')

        if (da_dict['daug_ph1']['type'] == 'affine'): # affine_transform(weightage =45%)
            param1D = da_dict['daug_ph1']['param1D']
            f.write("tcsh do_affine.tcsh {} {}""".format(fl_name,\
                                                        param1D))
            f.write('\n')
        # phase-2  data augmentation shell script

        phase2_daug = da_dict['daug_ph2']['type']

        match phase2_daug:

            case 'gain_inhom':
                axis   = da_dict['daug_ph2']['axis']
                window = da_dict['daug_ph2']['window']
                f.write("tcsh do_gain_inhomogenity.tcsh {} {} {}""".format(fl_name,\
                                                        axis, window))

            case 'zipper':
                axis  = da_dict['daug_ph2']['axis']
                width = da_dict['daug_ph2']['zip_width']
                f.write("tcsh do_zipper_noise.tcsh {} {} {}""".format(fl_name,\
                                                        axis, width))

            case 'add_noise':
                axis         = da_dict['daug_ph2']['axis']
                noise_level  = da_dict['daug_ph2']['noise_lvl']
                f.write("tcsh do_add_noise.tcsh {} {} {}""".format(fl_name,\
                                                        axis, noise_level))

            case 'contrast_var':
                shading         = da_dict['daug_ph2']['shading']
                if shading ==1:
                    f.write("tcsh do_contrast_variation_shading.tcsh {} """.format(fl_name))
                else:
                    f.write("tcsh do_contrast_variation_core.tcsh {} """.format(fl_name))
        f.write('\n')

    
    # create the edt data for all the masks 

    
    f.write("tcsh do_daug_weight.tcsh {}""".format(fl_name))

    f.write('\n')
    f.close()


def main():
    print('++ Python version  =', sys.version.split()[0])

    # ----- read in command line arguments
    args        = get_data_augment_args()

    # path for data augmentation folder
    da_path     = str(args.output_dir[0])
    # ... and strip any '/' at the right
    da_path     = da_path.rstrip('/')
    if len(da_path) == 0 :
        print("** ERROR: entered path was only '/', which is not allowed")
        sys.exit(3)

    # path for the dataset folder
    data_path   = str(args.input_dir[0])

    # number of copies of the dataset to be made 
    num_cp      = int(args.num_cp[0])
    
    # random seed number
    seed_num    = int(args.seed_num[0])

    # execution mode for set of scripts
    exec_mode   = str(args.exec_mode[0])
    # ... and make sure it is allowed
    tmp = is_valid_exec_mode(exec_mode)

    # verbosity level
    verb        = int(args.verb[0])

    if(os.path.isdir(da_path) == True):
        print('** ERROR Status msg : Data augmentation folder already exists')
        print('   Please change the data_augmentation directory name and retry')
        sys.exit(1) 

    # put the random seed in place
    print('++ Random seed_num =', seed_num)
    random.seed(seed_num)

    # ----- get name of augmentation script directory:
    dir_aug_scr = get_aug_script_dir()
    print("++ Found augmentation scripts dir:", dir_aug_scr)

    # ----- start initializing augmentation

    print("++ Run make_copies_data_aug.tcsh (num_cp = {})...".format(num_cp))
    cmd  = '''tcsh make_copies_data_aug.tcsh {param1} {param2} {param3}
    '''.format(param1 = data_path, param2 = da_path, param3 = num_cp)

    com  = ab.shell_com(cmd, capture=1)
    stat = com.run()
    # print the status of preparing the data_augmentation folder
    if (stat == 0):
        print("++ Status msg : Data augmentation folder succesfully created")
    else :
        print("** ERROR: Status msg : Data augmentation folder not created")
        print("   Failed command was:")
        print("   " + cmd)
        print("   Failed st err output:")
        print('-'*50)
        print('\n'.join(com.se))
        print('-'*50)
        sys.exit(1)
    
    # read copies of dataset
    orig_path_str = os.path.join(da_path, 'orig', '*.nii.gz')

    # list of copies of dataset
    orig_data_list = glob.glob(orig_path_str)
    orig_data_list.sort()

    # output directory, scripts subdirectory
    odir_scripts = os.path.join(da_path, 'scripts')

    print("++ Create master script for augmentation....")
    master_fl = scr_swarm
    master_fl_path_str = os.path.join(odir_scripts, master_fl)
    fl        = open(master_fl_path_str, "w")
    #fl.write("#!/bin/tcsh")
    #fl.write('\n \n \n')
    for fl_name in orig_data_list:
        # condition to check whether to daug or not
        fl_basename = os.path.basename(fl_name)
        print("++ Processing :",fl_basename)
        name = fl_basename.split('.')[0]
        print("   dset root:", name)
        log_fl = "log_" + name + ".txt"
        print("   log file :", log_fl)
        fl_num = int(re.search(r'\d+', fl_basename).group(0))
        if ((fl_num%1000) ==0):
            print('++ Status msg : Original copy of dset: no data augmentation')
            daug = 0 # flag to depict data_augmentation
            daug_dict = {} # empty dictionary
            print("fl_name =",fl_name)
            write_script(daug_dict,fl_name,daug)

        else: 
            print('++ Status msg : do augmentation')
            daug_dict = get_daug_dict(fl_basename)

            # write scripts based on the dictionary for each dataset
            daug = 1 # flag to depict data_augmentation
            write_script(daug_dict,fl_name,daug)
            print("daug_dict = ",daug_dict)
            list_daug.append(daug_dict)
            #|& tee logs/log_sub-001001_orig.txt
        fl.write("tcsh -x {}.tcsh |& tee ../logs/{}""".format(fl_basename.split('.')[0],log_fl))
        fl.write('\n')
            # list of dict
    fl.close()
            
    # Serializing json, which is now output to scripts dir in output dir
    json_object = json.dumps(list_daug, indent=4)
    with open(odir_scripts + "/daug.json", "w") as outfile:
        outfile.write(json_object)

    # copy the augmentation scripts to the output dir
    if 1 :
        tmp = copy_all_aug_script(dir_aug_scr, odir_scripts)

    # make a swarm script
    if 1 :
        # write the swarm script...
        full_path_swarm = odir_scripts + "/" + run_swarm
        fff = open(full_path_swarm, 'w')
        fff.write(cmd_swarm)
        fff.close()

    # (maybe) execute augmentation
    mode_str = "data augmentation in {mode} mode".format(mode=exec_mode)
    if exec_mode == 'swarm' :
        print("++ Run {}".format(mode_str))
        # for now, need to go to scripts dir bc of relative path for logs dir
        cmd  = '''cd {odir}; tcsh {file}'''.format(odir=odir_scripts,
                                                   file=full_path_swarm)
        com  = ab.shell_com(cmd, capture=1)
        stat = com.run()

        if stat :
            print("** ERROR: failed {}".format(mode_str))
            print("   Failed command was:")
            print("   " + cmd)
            print("   Failed st err output:")
            print('-'*50)
            print('\n'.join(com.se))
            print('-'*50)
            sys.exit(1)
    elif exec_mode == 'shell' :
        print("++ Run augmentation in {mode} mode".format(mode=exec_mode))
        # for now, need to go to scripts dir bc of relative path for logs dir
        cmd  = '''cd {odir}; tcsh {file}'''.format(odir=odir_scripts,
                                                   file=master_fl_path_str)
        com  = ab.shell_com(cmd, capture=1)
        stat = com.run()

        if stat :
            print("** ERROR: failed {}".format(mode_str))
            print("   Failed command was:")
            print("   " + cmd)
            print("   Failed st err output:")
            print('-'*50)
            print('\n'.join(com.se))
            print('-'*50)
            sys.exit(1)
    elif exec_mode == 'None' :
        print("++ Both master and swarm scripts ready to be run:")
    else:
        print("** ERROR: unrecognized exec_mode: ", exec_mode)

    return 0

# ===========================================================================

if __name__ == "__main__" :

    tmp = main()

    # exit ok
    sys.exit(0)
