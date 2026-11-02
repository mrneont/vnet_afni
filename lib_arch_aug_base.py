#!/usr/bin/env python3

"""Step-by-step preparation and execution of V-net data augmentation."""

import copy
import json
import os
import random
import shlex
import shutil

from afnipy    import afni_base          as ab
from vnet_afni import lib_arch_aug_defs  as DEF
from vnet_afni import lib_arch_vnet_base as LAVB

# ============================================================================

def quote_tcsh(value):
    """Quote one literal command argument for tcsh.

    Escape quote and history characters.  Reject line breaks and NULs,
    which cannot be placed safely in the generated command lines.
    """

    text = str(value)

    for char in ['\n', '\r', '\x00']:
        if char in text:
            msg = 'Newlines and NUL chars are not supported '
            msg+= 'in script arguments'
            raise ValueError(msg)

    text = text.replace('!', r'\!')
    text = text.replace("'", "'\\''")
    text = "'" + text + "'"

    return text


def command_tcsh(argv):
    """Build one tcsh command from a list of literal arguments.

    Quote each argument separately so paths and parameters retain their
    boundaries when the generated script is executed.
    """

    cmd = ''

    for arg in argv:
        if cmd:
            cmd += ' '

        cmd += quote_tcsh(arg)

    return cmd


# ============================================================================

class MainObj:
    """Store augmentation settings and run the preparation stages in order."""

    def __init__(self, user_inobj=None, args_orig=None):
        """Initialize processing state and run the requested augmentation.

        Without an input object, initialize attributes without running any
        stages.  Otherwise, stop at the first failed stage and retain its
        return value in self.status.
        """

        self.status     = 0
        self.user_inobj = user_inobj
        self.args_orig  = list(args_orig or [])

        for key, val in DEF.DOPTS.items():
            setattr(self, key, copy.deepcopy(val))

        self.split_tree     = None
        self.input_pairs    = []
        self.plans          = []
        self.script_sources = {}
        self.jobs           = []
        self.odir_scripts   = None
        self.weight_command = None

        # Run each stage only after the preceding stage succeeds.
        if user_inobj is not None:
            try:
                self.status = self.load_from_inopts()
                if self.status:
                    return

                self.status = self.basic_setup()
                if self.status:
                    return

                self.status = self.find_input_pairs()
                if self.status:
                    return

                self.status = self.make_plans()
                if self.status:
                    return

                self.status = self.find_aug_scripts()
                if self.status:
                    return

                self.status = self.make_outdir()
                if self.status:
                    return

                self.status = self.copy_dsets()
                if self.status:
                    return

                self.status = self.copy_aug_scripts()
                if self.status:
                    return

                self.status = self.write_scripts()
                if self.status:
                    return

                self.status = self.write_plan_and_log()
                if self.status:
                    return

                self.status = self.execute_scripts()
                if self.status:
                    return

            except (OSError, ValueError) as exc:
                ab.EP1(str(exc))
                self.status = -1


    def load_from_inopts(self):
        """Copy supplied input values over the processing defaults.

        Leave the default in place when the input value is None.  Copy
        values so the input and processing objects do not share containers.
        """

        for key in DEF.DOPTS:
            val = getattr(self.user_inobj, key, None)

            if val is not None:
                setattr(self, key, copy.deepcopy(val))

        return 0


    def basic_setup(self):
        """Check control values, output location and execution tools.

        Convert directory names to absolute paths and the weight option to
        a boolean.  Return -1 on failure before creating output files.
        """

        if not self.indir or not self.outdir:
            ab.EP1('Need to provide both -indir and -outdir')
            return -1

        if self.num_cp < 1 or self.verb < 0:
            ab.EP1('Need -num_cp >= 1 and -verb >= 0')
            return -1

        if self.exec_mode not in DEF.LIST_exec_mode:
            msg = 'Invalid -exec_mode; choose from: '
            msg+= DEF.STR_exec_mode
            ab.EP1(msg)
            return -1

        if str(self.do_wtds).lower() not in DEF.DICT_yes_no:
            ab.EP1('-do_wtds must be Yes or No')
            return -1

        self.do_wtds    = DEF.DICT_yes_no[str(self.do_wtds).lower()]
        self.indir      = os.path.abspath(self.indir)
        self.outdir     = os.path.abspath(self.outdir)
        self.script_dir = os.path.abspath(self.script_dir)

        if os.path.lexists(self.outdir):
            msg = 'Output path exists already; choose a new -outdir: '
            msg+= self.outdir
            ab.EP1(msg)
            return -1

        if self.exec_mode != 'None' and shutil.which('tcsh') is None:
            ab.EP1('Cannot execute scripts: tcsh is not on PATH')
            return -1

        if self.exec_mode == 'swarm' and shutil.which('swarm') is None:
            ab.EP1('Cannot submit scripts: swarm is not on PATH')
            return -1

        self.odir_scripts = os.path.join(self.outdir, 'scripts')

        if self.verb:
            ab.IP('Planning seed: {}'.format(self.seed_num))

        return 0


    def find_input_pairs(self):
        """Load orig-mask pairs checked by the shared V-net split object.

        ArchSplitTree checks the directories and sorts and matches their
        filenames.  Convert its paired entries to augmentation input
        records, retaining the common extension and full subject stem.
        """

        BAD_RETURN = -1

        # Use the same split validation and pairing as V-net training.
        self.split_tree = LAVB.ArchSplitTree(
            self.indir,
            has_mask = True,
            has_wtds = False,
            verb     = self.verb
        )

        if self.split_tree.status:
            ab.EP1('Failed to load input split tree: ' + self.indir)
            return BAD_RETURN

        num_pairs = self.split_tree.count_subdir_files('orig')

        if not num_pairs:
            ab.EP1('No anatomical datasets in input split: ' + self.indir)
            return BAD_RETURN

        dir_orig = self.split_tree.get_subdir_path('orig')
        dir_mask = self.split_tree.get_subdir_path('mask')

        # Shared sorted lists have matching filenames at each index.
        for index in range(num_pairs):
            fname_orig = self.split_tree.all_dset['orig'][index]
            fname_mask = self.split_tree.all_dset['mask'][index]

            path_orig = os.path.join(dir_orig, fname_orig)
            path_mask = os.path.join(dir_mask, fname_mask)

            if not os.path.isfile(path_orig) or not os.path.isfile(path_mask):
                msg = 'Expected anatomical and mask files: '
                msg+= '{} and {}'.format(path_orig, path_mask)
                ab.EP1(msg)
                return BAD_RETURN

            # Extract the stem and extension for naming augmented copies.
            ext = None

            for candidate in DEF.LIST_nii_ext:
                suffix = '_orig' + candidate

                if fname_orig.endswith(suffix):
                    ext = candidate
                    break

            if ext is None:
                msg = 'Expected an *_orig.nii or *_orig.nii.gz filename: '
                msg+= fname_orig
                ab.EP1(msg)
                return BAD_RETURN

            suffix = '_orig' + ext
            stem = fname_orig[:-len(suffix)]

            if not stem:
                ab.EP1('Empty input stem: ' + fname_orig)
                return BAD_RETURN

            pair = {
                'stem'        : stem,
                'source_orig' : path_orig,
                'source_mask' : path_mask,
                'ext_orig'    : ext,
                'ext_mask'    : ext,
            }

            self.input_pairs.append(pair)

        return 0


    def make_affine_params(self, rng):
        """Draw two distinct affine components and return their 12 values.

        Start from identity scales and use the shared parameter ranges.
        Draw all random values from the supplied generator.
        """

        params = DEF.LIST_affine_identity[:]
        kinds = rng.sample(DEF.LIST_affine_types, 2)

        for kind in kinds:

            if kind == 'rotation':
                axis  = rng.choice(DEF.LIST_axis_xyz)
                value = 0

                while value == 0:
                    value = rng.randint(*DEF.RANGE_rotation)

                index = DEF.DICT_rotation_index[axis]
                params[index] = value

            elif kind == 'shift':
                axis  = rng.choice(DEF.LIST_axis_xyz)
                value = rng.choice(DEF.LIST_shift)

                index = DEF.LIST_axis_xyz.index(axis)
                params[index] = value

            elif kind == 'scale':
                value = rng.uniform(*DEF.RANGE_scale)
                value = round(value, 1)

                params[6:9] = [value, value, value]

            elif kind == 'shear':
                axis  = rng.choice(DEF.LIST_axis_xyz)
                value = 0

                while value == 0:
                    value = rng.uniform(*DEF.RANGE_shear)
                    value = round(value, 2)

                index = DEF.DICT_shear_index[axis]
                params[index] = value

        return params


    def make_phase_plan(self, rng):
        """Select three augmentation phases and draw their parameters.

        Choose the phase types before drawing any parameters, keeping the
        random draw order stable for a given seed.  Return a new dictionary
        without changing shared definitions or global random state.
        """

        phases = {}

        # Choose all three phase types before drawing their parameters.
        for phase_name in DEF.LIST_phases:
            phase_types = DEF.DICT_phase_types[phase_name]
            types       = list(phase_types.keys())
            weights     = list(phase_types.values())
            choice      = rng.choices(types, weights)

            phases[phase_name] = {'type': choice[0]}

        ph0 = phases['daug_ph0']
        ph1 = phases['daug_ph1']
        ph2 = phases['daug_ph2']

        # Set parameters for the optional refacing phase.
        if ph0['type'] == 'reface':
            ph0['mode'] = DEF.MODE_reface

        # Set parameters for the geometric phase.
        if ph1['type'] == 'gibbs':
            ph1['radius'] = rng.randint(*DEF.RANGE_gibbs)

        elif ph1['type'] == 'affine':
            ph1['param1D'] = self.make_affine_params(rng)

        # Set parameters for the intensity or artifact phase.
        if ph2['type'] == 'gain_inhom':
            choice = rng.choices(DEF.LIST_axis_ijk,
                                 DEF.LIST_gain_axis_weights)
            window = rng.uniform(*DEF.RANGE_gain_window)

            ph2['axis'] = choice[0]
            ph2['window'] = round(window, 1)

        elif ph2['type'] == 'zipper':
            ph2['axis'] = rng.choice(DEF.LIST_axis_ijk)

        elif ph2['type'] == 'add_noise':
            ph2['noise_lvl'] = rng.choice(DEF.LIST_noise_level)

        elif ph2['type'] == 'contrast_var':
            ph2['shading'] = rng.randint(0, 1)
            ph2['dim_fac'] = DEF.CONTRAST_dim_fac

            if ph2['shading']:
                ph2['axis'] = DEF.CONTRAST_axis
                ph2['window'] = DEF.CONTRAST_window

        return phases


    def make_plans(self):
        """Build one output plan per subject and requested copy index.

        Copy 000 has empty augmentation phases.  Other copies receive
        random phase plans.  Store output paths with each plan and reject
        colliding labels before any files are copied.
        """

        rng = random.Random(self.seed_num)
        output_names = set()

        for pair in self.input_pairs:
            for index in range(self.num_cp):
                label = '{}{:03d}'.format(pair['stem'], index)

                if label in output_names:
                    ab.EP1('Copy labels collide: ' + label)
                    return -1

                output_names.add(label)

                orig_name = label + '_orig' + pair['ext_orig']
                mask_name = label + '_mask' + pair['ext_mask']
                wtds_name = label + '_wtds.nii.gz'

                plan = dict(pair)
                plan['copy_index'] = index
                plan['label'] = label
                plan['fl_basename'] = orig_name
                plan['orig'] = os.path.join(self.outdir, 'orig', orig_name)
                plan['mask'] = os.path.join(self.outdir, 'mask', mask_name)
                plan['wtds'] = None

                if self.do_wtds:
                    plan['wtds'] = os.path.join(self.outdir, DEF.DIR_wtds,
                                                wtds_name)

                if index:
                    phases = self.make_phase_plan(rng)
                    plan.update(phases)

                else:
                    for phase_name in DEF.LIST_phases:
                        plan[phase_name] = {'type': 'empty'}

                self.plans.append(plan)

        return 0


    def get_aug_key(self, phase):
        """Return the script lookup key for a selected augmentation phase.

        Resolve contrast variation to either the core or shading script.
        Other phase types already match their script lookup keys.
        """

        key = phase['type']

        if key == 'contrast_var':
            if phase['shading']:
                return 'contrast_shading'
            else:
                return 'contrast_core'

        return key


    def find_aug_scripts(self):
        """Find script sources and check that every selected script exists.

        Prepare the weight command from a local helper when available, or
        use its executable name on PATH.  Check PATH immediately for serial
        execution; preparation and swarm jobs resolve it when they run.
        """

        required = set()

        # Collect the scripts used by the planned augmentation phases.
        for plan in self.plans:
            for phase_name in DEF.LIST_phases:
                phase = plan[phase_name]

                if phase['type'] == 'empty':
                    continue

                key = self.get_aug_key(phase)
                required.add(key)

        for key, fname in DEF.DICT_aug_script.items():
            path = os.path.join(self.script_dir, fname)

            if os.path.isfile(path):
                self.script_sources[fname] = path

            elif key in required:
                msg = 'Cannot find augmentation script: '
                msg+= '{} (see -script_dir)'.format(path)
                ab.EP1(msg)
                return -1

        # Use the same weight-helper interface as lib_arch_vnet_wtds.
        if self.do_wtds:
            path = os.path.join(self.script_dir, DEF.PROG_mask_to_wtds)

            if os.path.isfile(path):
                self.script_sources[DEF.PROG_mask_to_wtds] = path

                helper = os.path.join(self.odir_scripts,
                                      DEF.PROG_mask_to_wtds)
                self.weight_command = ['tcsh', '-f', helper]

            else:
                self.weight_command = [DEF.PROG_mask_to_wtds]

                if self.exec_mode == 'shell':
                    helper = shutil.which(DEF.PROG_mask_to_wtds)

                    if helper is None:
                        msg = 'Cannot find {} on PATH; '
                        msg = msg.format(DEF.PROG_mask_to_wtds)
                        msg+= 'use -do_wtds No to defer weights'
                        ab.EP1(msg)
                        return -1

        return 0


    def make_outdir(self):
        """Create a new output root and its dataset, script and log folders.

        Create wtds/ only when weight generation is requested.  Let an
        existing output root raise an error, including a creation race.
        """

        os.makedirs(self.outdir)

        subdirs = DEF.LIST_subdirs[:]

        if self.do_wtds:
            subdirs.append(DEF.DIR_wtds)

        for subdir in subdirs:
            path = os.path.join(self.outdir, subdir)
            os.mkdir(path)

        return 0


    def copy_dsets(self):
        """Copy each source anatomy and mask to the paths in its plan.

        Use separate output files for every copy so augmentation never
        writes to the source datasets.
        """

        for plan in self.plans:
            shutil.copy2(plan['source_orig'], plan['orig'])
            shutil.copy2(plan['source_mask'], plan['mask'])

        return 0


    def copy_aug_scripts(self):
        """Copy the discovered helper scripts without changing their bytes.

        Place them in the output scripts directory so generated jobs can
        refer to these copies by absolute path.
        """

        for fname, source in self.script_sources.items():
            target = os.path.join(self.odir_scripts, fname)
            shutil.copy2(source, target)

        return 0


    def get_phase_command(self, plan, phase):
        """Build the argument list for one augmentation script call.

        Supply the anatomy and mask explicitly as both inputs and outputs.
        The scripts copy inputs into isolated workdirs before writing the
        augmented results.  Keep each parameter flag beside its value.
        """

        key = self.get_aug_key(phase)
        script_name = DEF.DICT_aug_script[key]
        script_path = os.path.join(self.odir_scripts, script_name)

        # Each script receives the same four dataset options.
        cmd = ['tcsh', '-f', script_path]
        cmd += ['-inset_orig', plan['orig']]
        cmd += ['-inset_mask', plan['mask']]
        cmd += ['-prefix_orig', plan['orig']]
        cmd += ['-prefix_mask', plan['mask']]

        # Add only the parameters accepted by the selected script.
        if key == 'reface':
            cmd += ['-param_mode', phase['mode']]

        elif key == 'gibbs':
            cmd += ['-param_range', phase['radius']]

        elif key == 'affine':
            cmd += ['-param_affine']
            cmd += phase['param1D']

        elif key == 'gain_inhom':
            cmd += ['-param_axis', phase['axis']]
            cmd += ['-param_window', phase['window']]

        elif key == 'zipper':
            cmd += ['-param_axis', phase['axis']]

        elif key == 'add_noise':
            cmd += ['-param_noise_level', phase['noise_lvl']]

        elif key == 'contrast_core':
            cmd += ['-param_dim_fac', phase['dim_fac']]

        elif key == 'contrast_shading':
            cmd += ['-param_dim_fac', phase['dim_fac']]
            cmd += ['-param_axis', phase['axis']]
            cmd += ['-param_window', phase['window']]

        return cmd


    def write_tcsh(self, path, commands):
        """Write executable tcsh commands with a status check after each one.

        Exit with the first failed command's status.  When every command
        succeeds, exit with status 0.
        """

        with open(path, 'w') as out:
            out.write('#!/bin/tcsh -f\n')
            out.write('# Generated by arch_augment.py\n\n')

            for cmd in commands:
                out.write(cmd + '\n')
                out.write('set step_status = $status\n')
                out.write('if ( $step_status != 0 ) exit $step_status\n\n')

            out.write('exit 0\n')

        os.chmod(path, 0o755)


    def write_scripts(self):
        """Write per-copy jobs, the swarm command list and both runners.

        Put augmentation commands in phase order, followed by the optional
        weight command.  Give each job a separate log and record its script
        and log paths in self.jobs.
        """

        master_lines = []

        for plan in self.plans:
            commands = []

            # Add each selected augmentation in phase order.
            for phase_name in DEF.LIST_phases:
                phase = plan[phase_name]

                if phase['type'] == 'empty':
                    continue

                argv = self.get_phase_command(plan, phase)
                cmd = command_tcsh(argv)
                commands.append(cmd)

            # Make weights from the final mask after all augmentation steps.
            if self.do_wtds:
                argv = self.weight_command[:]
                argv+= [plan['mask']]
                argv+= [plan['wtds']]

                cmd = command_tcsh(argv)
                commands.append(cmd)

            script_name = plan['label'] + '_orig.tcsh'
            script = os.path.join(self.odir_scripts, script_name)

            log_name = 'log_' + plan['label'] + '_orig.txt'
            logfile = os.path.join(self.outdir, 'logs', log_name)

            self.write_tcsh(script, commands)
            self.jobs.append({'script': script, 'log': logfile})

            # Redirect each job directly to its log, retaining exit status.
            argv = ['tcsh', '-f', script]
            cmd = command_tcsh(argv)
            cmd += ' >& '
            cmd += quote_tcsh(logfile)

            master_lines.append(cmd)

        # Write one independent job per line for swarm.
        master_path = os.path.join(self.odir_scripts, DEF.FNAME_master)

        with open(master_path, 'w') as out:
            for cmd in master_lines:
                out.write(cmd + '\n')

        # Write the serial runner with a status check after each job.
        serial_path = os.path.join(self.odir_scripts, DEF.FNAME_serial)
        self.write_tcsh(serial_path, master_lines)

        # Build the swarm submission command with the shared resource options.
        log_path = os.path.join(self.outdir, 'logs')
        swarm_path = os.path.join(self.odir_scripts, DEF.FNAME_swarm)

        argv = ['swarm']
        argv += ['-f', master_path]
        argv += DEF.LIST_swarm_opts
        argv += ['--logdir=' + log_path]

        cmd = command_tcsh(argv)
        self.write_tcsh(swarm_path, [cmd])

        return 0


    def write_plan_and_log(self):
        """Save the complete plan as JSON and record the original command.

        Include the version and planning seed in the command log so the
        prepared output can be traced to its inputs and settings.
        """

        plan_path = os.path.join(self.odir_scripts, DEF.FNAME_plan)
        log_path = os.path.join(self.outdir, DEF.FNAME_cmd)

        with open(plan_path, 'w') as out:
            json.dump(self.plans, out, indent=4)
            out.write('\n')

        cmd_parts = []

        for arg in self.args_orig:
            cmd_parts.append(shlex.quote(arg))

        cmd = ' '.join(cmd_parts)

        with open(log_path, 'w') as out:
            out.write(DEF.g_version + '\n')
            out.write(cmd + '\n')
            out.write('seed_num = {}\n'.format(self.seed_num))

        return 0


    def execute_scripts(self):
        """Leave jobs prepared, run them serially, or submit them to swarm.

        Select the runner from exec_mode and execute it with ab.shell_com.
        Capture its output and report a failed command with its stderr.
        Swarm success indicates submission, not completion of its jobs.
        """

        if self.exec_mode == 'None':
            if self.verb:
                msg = 'Prepared {} copies; scripts are in: '
                msg = msg.format(len(self.plans))
                msg+= self.odir_scripts
                ab.IP(msg)

            return 0

        if self.exec_mode == 'shell':
            fname = DEF.FNAME_serial
        else:
            fname = DEF.FNAME_swarm

        if self.verb:
            msg = 'Run augmentation in {} mode'.format(self.exec_mode)
            ab.IP(msg)

        path = os.path.join(self.odir_scripts, fname)

        # Change directory in the child shell, leaving Python's cwd alone.
        cmd = 'cd {} && '.format(shlex.quote(self.odir_scripts))
        cmd+= 'tcsh -f {}'.format(shlex.quote(path))

        com  = ab.shell_com(cmd, capture=1)
        stat = com.run()

        if self.verb or stat:
            for line in com.so:
                print(line)

        if stat:
            log_path = os.path.join(self.outdir, 'logs')

            msg = 'Augmentation {} failed (status {}); '
            msg = msg.format(self.exec_mode, stat)
            msg+= 'inspect logs in ' + log_path
            ab.EP1(msg)

            print('Failed command:')
            print(cmd)

            for line in com.se:
                print(line)

            return -1

        return 0
