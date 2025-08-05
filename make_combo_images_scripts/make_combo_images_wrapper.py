import sys


# on most systems, this should come from regular AFNI install
sys.path.append('/Users/narayanaswamyy2/abin')
from afnipy import afni_base as ab
import os 
import argparse      as argp
import json
import random
import numpy as np
import os
import glob
import re #regular expression


# ============================================================================
# program version and brief notes on updates

version = '1.1' # adding in more help files

# ============================================================================

# default options and definitions

# default values for program opts
DEF = {
        'verb'          : 1,
        'seed_num'      : 42,
        'exec_mode'     : 'None',
        'pred_mask_dir' : '',
        'output_dir'    : '',
}

# list of allowed execution modes
LIST_exec_mode = ['None', 'swarm', 'shell']
STR_exec_mode  = ', '.join(LIST_exec_mode)


# default swarm script
scr_swarm = 'master_script.tcsh'
master_fl = scr_swarm
run_swarm = 'run_swarm.tcsh'
cmd_swarm = """#!/bin/tcsh

swarm                                                              \\
    -f  {scr_swarm}                                                \\
    --module afni                                                  \\
    --partition=norm,quick                                         \\
    --threads-per-process=2                                        \\
    --gb-per-process=3                                             \\
    --time=00:30:00                                                \\
    --logdir={cdir_log}                                            \\
    --job-name=job_{cmd}                                           \\
    --merge-output                                                 \\
    --usecsh
""".format(cmd='combo', scr_swarm=scr_swarm, cdir_log='../logs')

def is_valid_exec_mode(mode):
    """Is the given string 'mode' an allowed execution mode?  Exit on
failure."""

    if not(mode in LIST_exec_mode) :
        print("** ERROR: this is not a valid execution mode: {}"
              "".format(mode))
        sys.exit(7)

    return 0


def get_combo_script_dir():
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


def get_make_combo_args():

    parser = argp.ArgumentParser(prog = 'make_combo_images_wrapper.py',
                                    formatter_class=argp.RawTextHelpFormatter)

    # pred_mask_dir is the data_path for the predicted masks(input dir)
    parser.add_argument("-pred_mask_dir", nargs=1,
                        default=[DEF['pred_mask_dir']],
                        help='(req) data_path for the predicted masks'
                        '(def: {})'.format(DEF['pred_mask_dir']))

    # data_dir is the data path for the dataset folder
    parser.add_argument("-output_dir", nargs=1,
                        default=[DEF['output_dir']],
                        help='(req) path for output dir, '
                        ' which will be populated '
                        'with these subdirectories: "mask", "orig", "scripts", '
                        '"logs", and, if executing, "edt" '
                        '(def: {})'.format(DEF['output_dir']))
    # random seed number (integer)
    parser.add_argument("-seed_num", nargs=1,
                        default=[DEF['seed_num']],
                        help='seed number for randomization of augmentation '
                        'steps '
                        '(def: {})'.format(DEF['seed_num']))

    # execution mode for make_combo_images script set
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
    if len(args.pred_mask_dir) == 0 :
        print("** ERROR: need to provide an '-input_dir ..'")
        sys.exit(8)
    if len(args.output_dir) == 0 :
        print("** ERROR: need to provide an '-output_dir ..'")
        sys.exit(8)
   

    return parser.parse_args()


def main():

   # ----- read in command line arguments
    args   = get_make_combo_args()
    

    pred_mask_path     = args.pred_mask_dir
    #print(" data augmentation folder is: {}".format(pred_mask_path))
    
    # path for data augmentation folder
    pred_mask_path     = str(pred_mask_path[0])
    
    # ... and strip any '/' at the right
    pred_mask_path     = pred_mask_path.rstrip('/')
    print('pred_mask_path  = ',pred_mask_path)
    if len(pred_mask_path) == 0 :
        print("** ERROR: entered path was only '/', which is not allowed")
        sys.exit(3)

    # random seed number
    seed_num    = int(args.seed_num[0])

    # execution mode for set of scripts
    exec_mode   = str(args.exec_mode[0])
    print('exec_mode = ',exec_mode)
    # ... and make sure it is allowed
    tmp = is_valid_exec_mode(exec_mode)

     # ----- get name of make_combo_images script directory:
    dir_combo_scr = get_combo_script_dir()
    print("++ Found make_combo_images scripts dir:", dir_combo_scr)

    print("++ Run make_combo_image_folders.tcsh  ...")
    cmd  = '''tcsh make_combo_image_folders.tcsh {param1} {param2} 
    '''.format(param1 = pred_mask_path, param2 = dir_combo_scr)
    print("++  make_combo_image_folders.tcsh finished ...")
    com  = ab.shell_com(cmd, capture=1)
    stat = com.run()
    # print the status of preparing the data_augmentation folder
    if (stat == 0):
        print("++ Status msg : make_combo_images folder succesfully created")
    else :
        print("** ERROR: Status msg : make_combo_images folder not created")
        print("   Failed command was:")
        print("   " + cmd)
        print("   Failed st err output:")
        print('-'*50)
        print('\n'.join(com.se))
        print('-'*50)
        sys.exit(1)

   # Get the basename (last part of the path)
    basename = os.path.basename(pred_mask_path)
    print(basename)

    # Create the combo images directory name
    make_combo_images_dir = f"make_combo_images_{basename}"
    print(make_combo_images_dir)

    
    parent_dir  = os.path.dirname(dir_combo_scr)
    print(" Found parent_dir:", parent_dir)

    odir_scripts = os.path.join(parent_dir,make_combo_images_dir, 'scripts')
    master_fl_path_str = os.path.join(odir_scripts, master_fl)
    print("master_fl_path_str=",master_fl_path_str)

    mode_str = "data augmentation in {mode} mode".format(mode=exec_mode)

    # make a swarm script
    if 1 :
        # write the swarm script...
        full_path_swarm = odir_scripts + "/" + run_swarm
        fff = open(full_path_swarm, 'w')
        fff.write(cmd_swarm)
        fff.close()

    print("full_path_swarm=",full_path_swarm)
    if exec_mode == 'swarm' :
        print("++ Run {}".format(mode_str))
        # for now, need to go to scripts dir bc of relative path for logs dir
        cmd  = '''cd {odir}; tcsh {file}'''.format(odir=odir_scripts,
                                                   file=full_path_swarm)
        com  = ab.shell_com(cmd, capture=1)
        stat = com.run()
        print("++ completed  {}".format(mode_str))
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
  

main()