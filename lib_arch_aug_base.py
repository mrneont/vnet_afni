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
        self.phase_sampling = {}

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

                self.status = self.setup_phase_sampling()
                if self.status:
                    return

                self.status = self.find_aug_scripts()
                if self.status:
                    return

                self.status = self.make_plans()
                if self.status:
                    return

                self.status = self.make_outdir()
                if self.status:
                    return

                self.status = self.write_plan_and_log()
                if self.status:
                    return

                self.status = self.copy_aug_scripts()
                if self.status:
                    return

                self.status = self.write_scripts()
                if self.status:
                    return

                self.status = self.copy_dsets()
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

        Convert directory names to absolute paths.  Return -1 on failure
        before creating output files.
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

        self.indir      = os.path.abspath(self.indir)
        self.outdir     = os.path.abspath(self.outdir)
        self.script_dir = os.path.abspath(self.script_dir)

        # Keep each path within a single generated command line.
        for path in [self.indir, self.outdir, self.script_dir]:
            for char in ['\n', '\r', '\x00']:
                if char in path:
                    ab.EP1('Newlines and NUL chars are not allowed in paths')
                    return -1

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

            for fname in [fname_orig, fname_mask]:
                for char in ['\n', '\r', '\x00']:
                    if char in fname:
                        msg = 'Newlines and NUL chars are not allowed '
                        msg+= 'in dataset filenames'
                        ab.EP1(msg)
                        return BAD_RETURN

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
                rbot  = DEF.RANGE_rotation[0]
                rtop  = DEF.RANGE_rotation[1]

                while value == 0:
                    value = rng.randint(rbot, rtop)

                index = DEF.DICT_rotation_index[axis]
                params[index] = value

            elif kind == 'shift':
                axis  = rng.choice(DEF.LIST_axis_xyz)
                value = rng.choice(DEF.LIST_shift)

                index = DEF.LIST_axis_xyz.index(axis)
                params[index] = value

            elif kind == 'scale':
                rbot  = DEF.RANGE_scale[0]
                rtop  = DEF.RANGE_scale[1]
                value = rng.uniform(rbot, rtop)
                value = round(value, 1)

                params[6:9] = [value, value, value]

            elif kind == 'shear':
                axis  = rng.choice(DEF.LIST_axis_xyz)
                value = 0
                rbot  = DEF.RANGE_shear[0]
                rtop  = DEF.RANGE_shear[1]

                while value == 0:
                    value = rng.uniform(rbot, rtop)
                    value = round(value, 2)

                index = DEF.DICT_shear_index[axis]
                params[index] = value

        return params


    def setup_phase_sampling(self):
        """Prepare each phase's choices and relative weights once per run.

        Keep the definition order so seeded sampling follows the existing
        random draw sequence.  Later copies reuse these prepared lists.
        """

        for phase_name in DEF.LIST_phases:
            phase_types = DEF.DICT_phase_types[phase_name]

            self.phase_sampling[phase_name] = {
                'types'   : list(phase_types.keys()),
                'weights' : list(phase_types.values()),
            }

        return 0


    def make_phase_plan(self, rng):
        """Select three augmentation phases and draw their parameters.

        Choose the phase types before drawing any parameters, keeping the
        random draw order stable for a given seed.  Return a new dictionary
        without changing shared definitions or global random state.
        """

        phases = {}

        # Choose all three phase types before drawing their parameters.
        for phase_name in DEF.LIST_phases:
            phase_sampling = self.phase_sampling[phase_name]
            types          = phase_sampling['types']
            weights        = phase_sampling['weights']
            choice         = rng.choices(types, weights)

            phases[phase_name] = {'type': choice[0]}

        ph0 = phases['daug_ph0']
        ph1 = phases['daug_ph1']
        ph2 = phases['daug_ph2']

        # Set parameters for the optional refacing phase.
        if ph0['type'] == 'reface':
            ph0['mode'] = DEF.MODE_reface

        # Set parameters for the geometric phase.
        if ph1['type'] == 'gibbs':
            rbot = DEF.RANGE_gibbs[0]
            rtop = DEF.RANGE_gibbs[1]
            ph1['radius'] = rng.randint(rbot, rtop)

        elif ph1['type'] == 'affine':
            ph1['param1D'] = self.make_affine_params(rng)

        # Set parameters for the intensity or artifact phase.
        if ph2['type'] == 'gain_inhom':
            choice = rng.choices(DEF.LIST_axis_ijk,
                                 DEF.LIST_gain_axis_weights)
            rbot   = DEF.RANGE_gain_window[0]
            rtop   = DEF.RANGE_gain_window[1]
            window = rng.uniform(rbot, rtop)

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

                plan = dict(pair)
                plan['copy_index'] = index
                plan['label'] = label
                plan['fl_basename'] = orig_name
                plan['orig'] = os.path.join(self.outdir, 'orig', orig_name)
                plan['mask'] = os.path.join(self.outdir, 'mask', mask_name)
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
        """Check the complete augmentation script set before making plans.

        Record each source path once for the later copying stage.  Require
        all supported scripts, regardless of this run's random selections.
        """

        for script_name in DEF.DICT_aug_script.values():
            script_path = os.path.join(self.script_dir, script_name)

            if not os.path.isfile(script_path):
                msg = 'Cannot find augmentation script: '
                msg+= '{} (see -script_dir)'.format(script_path)
                ab.EP1(msg)
                return -1

            self.script_sources[script_name] = script_path

        return 0


    def make_outdir(self):
        """Create a new output root and its dataset, script and log folders.

        Let an existing output root raise an error, including a creation
        race after the earlier output-path check.
        """

        os.makedirs(self.outdir)

        for subdir in DEF.LIST_subdirs:
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
        """Build one augmentation command by joining explicit option pieces.

        Supply anatomy and mask as both inputs and outputs.  Quote paths
        inline, convert numeric parameters to strings, and keep each flag
        beside its value.  Return the completed command string.
        """

        augmentation_type = self.get_aug_key(phase)
        augmentation_script_name = DEF.DICT_aug_script[augmentation_type]
        augmentation_script_path = os.path.join(
            self.odir_scripts, augmentation_script_name
        )

        # Quote paths inline, including tcsh's history character.
        quoted_script_path = shlex.quote(augmentation_script_path)
        quoted_script_path = quoted_script_path.replace('!', r'\!')
        quoted_orig_path   = shlex.quote(plan['orig']).replace('!', r'\!')
        quoted_mask_path   = shlex.quote(plan['mask']).replace('!', r'\!')

        # Each script receives the same four dataset options.
        phase_command_parts = ['tcsh', '-f', quoted_script_path]
        phase_command_parts += ['-inset_orig', quoted_orig_path]
        phase_command_parts += ['-inset_mask', quoted_mask_path]
        phase_command_parts += ['-prefix_orig', quoted_orig_path]
        phase_command_parts += ['-prefix_mask', quoted_mask_path]

        # Add only the parameters accepted by the selected script.
        if augmentation_type == 'reface':
            phase_command_parts += ['-param_mode', phase['mode']]

        elif augmentation_type == 'gibbs':
            phase_command_parts += ['-param_range', str(phase['radius'])]

        elif augmentation_type == 'affine':
            phase_command_parts += ['-param_affine']

            for value in phase['param1D']:
                phase_command_parts.append(str(value))

        elif augmentation_type == 'gain_inhom':
            phase_command_parts += ['-param_axis', phase['axis']]
            phase_command_parts += ['-param_window', str(phase['window'])]

        elif augmentation_type == 'zipper':
            phase_command_parts += ['-param_axis', phase['axis']]

        elif augmentation_type == 'add_noise':
            phase_command_parts += [
                '-param_noise_level', str(phase['noise_lvl'])
            ]

        elif augmentation_type == 'contrast_core':
            phase_command_parts += ['-param_dim_fac', str(phase['dim_fac'])]

        elif augmentation_type == 'contrast_shading':
            phase_command_parts += ['-param_dim_fac', str(phase['dim_fac'])]
            phase_command_parts += ['-param_axis', phase['axis']]
            phase_command_parts += ['-param_window', str(phase['window'])]

        phase_command = ' '.join(phase_command_parts)

        return phase_command


    def write_tcsh(self, script_path, command_strings):
        """Write tcsh command strings with a status check after each one.

        Exit with the first failed command's status.  When every command
        succeeds, exit with status 0.
        """

        with open(script_path, 'w') as out:
            out.write('#!/bin/tcsh -f\n')
            out.write('# Generated by arch_augment.py\n\n')

            for command_string in command_strings:
                out.write(command_string + '\n')
                out.write('set step_status = $status\n')
                out.write('if ( $step_status != 0 ) exit $step_status\n\n')

            out.write('exit 0\n')

        os.chmod(script_path, 0o755)


    def write_scripts(self):
        """Write per-copy jobs, the swarm command list and both runners.

        Put augmentation commands in phase order.  Give each job a separate
        log and record its script and log paths in self.jobs.
        """

        master_commands = []

        for plan in self.plans:
            dataset_commands = []

            # Add each selected augmentation in phase order.
            for phase_name in DEF.LIST_phases:
                phase = plan[phase_name]

                if phase['type'] == 'empty':
                    continue

                phase_command = self.get_phase_command(plan, phase)
                dataset_commands.append(phase_command)

            dataset_script_name = plan['label'] + '_orig.tcsh'
            dataset_script_path = os.path.join(
                self.odir_scripts, dataset_script_name
            )

            dataset_log_name = 'log_' + plan['label'] + '_orig.txt'
            dataset_log_path = os.path.join(
                self.outdir, 'logs', dataset_log_name
            )

            self.write_tcsh(dataset_script_path, dataset_commands)
            self.jobs.append({
                'script' : dataset_script_path,
                'log'    : dataset_log_path,
            })

            # Redirect each job directly to its log, retaining exit status.
            quoted_script_path = shlex.quote(dataset_script_path)
            quoted_script_path = quoted_script_path.replace('!', r'\!')
            quoted_log_path = shlex.quote(dataset_log_path)
            quoted_log_path = quoted_log_path.replace('!', r'\!')

            job_command_parts = ['tcsh', '-f', quoted_script_path]
            job_command_parts += ['>&', quoted_log_path]

            job_command = ' '.join(job_command_parts)
            master_commands.append(job_command)

        # Write one independent job per line for swarm.
        master_path = os.path.join(self.odir_scripts, DEF.FNAME_master)

        with open(master_path, 'w') as out:
            for job_command in master_commands:
                out.write(job_command + '\n')

        # Write the serial runner with a status check after each job.
        serial_path = os.path.join(self.odir_scripts, DEF.FNAME_serial)
        self.write_tcsh(serial_path, master_commands)

        # Build the swarm submission command with the shared resource options.
        log_path = os.path.join(self.outdir, 'logs')
        swarm_path = os.path.join(self.odir_scripts, DEF.FNAME_swarm)

        quoted_master_path = shlex.quote(master_path).replace('!', r'\!')
        quoted_log_path    = shlex.quote(log_path).replace('!', r'\!')

        swarm_command_parts = ['swarm']
        swarm_command_parts += ['-f', quoted_master_path]
        swarm_command_parts += DEF.LIST_swarm_opts
        swarm_command_parts += ['--logdir=' + quoted_log_path]

        swarm_command = ' '.join(swarm_command_parts)
        self.write_tcsh(swarm_path, [swarm_command])

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

        original_command_parts = []

        for original_argument in self.args_orig:
            original_command_parts.append(shlex.quote(original_argument))

        original_command = ' '.join(original_command_parts)

        with open(log_path, 'w') as out:
            out.write(DEF.g_version + '\n')
            out.write(original_command + '\n')
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
            runner_script_name = DEF.FNAME_serial
        else:
            runner_script_name = DEF.FNAME_swarm

        if self.verb:
            msg = 'Run augmentation in {} mode'.format(self.exec_mode)
            ab.IP(msg)

        runner_script_path = os.path.join(
            self.odir_scripts, runner_script_name
        )

        # Change directory in the child shell, leaving Python's cwd alone.
        runner_command_parts = ['cd', shlex.quote(self.odir_scripts)]
        runner_command_parts += [
            '&&', 'tcsh', '-f', shlex.quote(runner_script_path)
        ]

        runner_command = ' '.join(runner_command_parts)

        com  = ab.shell_com(runner_command, capture=1)
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
            print(runner_command)

            for line in com.se:
                print(line)

            return -1

        return 0
