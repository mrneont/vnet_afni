#!/usr/bin/env python

"""AFNI command-line option handling for arch_augment.py."""

import sys

from afnipy    import option_list       as OL
from afnipy    import afni_base         as ab
from vnet_afni import lib_arch_aug_defs as DEF

# ============================================================================

class InOpts:
    """Read typed inputs and check them before creating MainObj."""

    def __init__(self, argv=None):
        """Initialize input attributes and register valid command options.

        Keep unspecified inputs as None so MainObj can apply its defaults.
        Save a separate argument list for the command log.
        """

        self.status     = 0
        self.valid_opts = None
        self.user_opts  = None

        if argv is None:
            self.argv = list(sys.argv)
        else:
            self.argv = list(argv)

        self.args_orig = self.argv[:]

        self.indir      = None
        self.outdir     = None
        self.num_cp     = None
        self.seed_num   = None
        self.exec_mode  = None
        self.script_dir = None
        self.do_qc_imgs = None
        self.verb       = DEF.DOPTS['verb']

        # -----

        self.init_options()

    def init_options(self):
        """Register option names, argument counts and short help strings.

        Read parameter types in process_options(), after AFNI has checked
        the option names and the number of arguments.
        """

        self.valid_opts = OL.OptionList('valid opts')

        # Register terminal options separately from processing inputs.
        for name, helpstr in DEF.LIST_terminal_opts:
            self.valid_opts.add_opt(name, 0, [], helpstr=helpstr)

        for name, attr, otype, helpstr in DEF.LIST_opts:
            self.valid_opts.add_opt(name, 1, [], helpstr=helpstr)

        return 0


    def process_options(self):
        """Handle terminal options and read each processing input by type.

        Return 1 after a terminal option, 0 to continue processing, or -1
        when an option or its parameter cannot be interpreted.
        """

        self.valid_opts.check_special_opts(self.argv)

        if len(self.argv) <= 1 or '-help' in self.argv or '-h' in self.argv:
            print(DEF.g_help_string)
            return 1

        if '-ver' in self.argv:
            print(DEF.g_version)
            return 1

        if '-hist' in self.argv:
            print(DEF.g_history)
            return 1

        if '-show_valid_opts' in self.argv:
            self.valid_opts.show('', 1)
            return 1

        self.user_opts = OL.read_options(self.argv, self.valid_opts)
        uopts = self.user_opts
        if uopts is None:
            return -1

        # Read verbosity before processing the other inputs.
        val, err = uopts.get_type_opt(int, '-verb')
        if err:
            return -1

        if val is not None:
            self.verb = val

        # Process inputs in command-line order, including repeated options.
        err_base = 'Problem interpreting use of opt: '

        for opt in uopts.olist:

            if opt.name == '-indir':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    ab.EP1(err_base + opt.name)
                    return -1
                self.indir = val

            elif opt.name == '-outdir':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    ab.EP1(err_base + opt.name)
                    return -1
                self.outdir = val

            elif opt.name == '-num_cp':
                val, err = uopts.get_type_opt(int, '', opt=opt)
                if val is None or err:
                    ab.EP1(err_base + opt.name)
                    return -1
                self.num_cp = val

            elif opt.name == '-seed_num':
                val, err = uopts.get_type_opt(int, '', opt=opt)
                if val is None or err:
                    ab.EP1(err_base + opt.name)
                    return -1
                self.seed_num = val

            elif opt.name == '-exec_mode':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    ab.EP1(err_base + opt.name)
                    return -1
                self.exec_mode = val

            elif opt.name == '-script_dir':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    ab.EP1(err_base + opt.name)
                    return -1
                self.script_dir = val

            elif opt.name == '-do_qc_imgs':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    ab.EP1(err_base + opt.name)
                    return -1
                self.do_qc_imgs = val

            elif opt.name == '-verb':
                val, err = uopts.get_type_opt(int, '', opt=opt)
                if val is None or err:
                    ab.EP1(err_base + opt.name)
                    return -1
                self.verb = val

        return 0


    def check_options(self):
        """Check required inputs and the ranges of supplied control values.

        Leave filesystem checks and default application to MainObj.
        Return 0 on success or -1 when an input is invalid.
        """

        BAD_RETURN = -1

        if not self.indir:
            ab.EP1('missing -indir option')
            return BAD_RETURN

        if not self.outdir:
            ab.EP1('missing -outdir option')
            return BAD_RETURN

        if self.num_cp is not None and self.num_cp < 1:
            msg = '-num_cp must be at least 1 '
            msg+= '(including the original copy)'
            ab.EP1(msg)
            return BAD_RETURN

        if self.verb < 0:
            ab.EP1('-verb must be nonnegative')
            return BAD_RETURN

        if self.exec_mode is not None:
            if self.exec_mode not in DEF.LIST_exec_mode:
                msg = '-exec_mode must be one of: '
                msg+= DEF.STR_exec_mode
                ab.EP1(msg)
                return BAD_RETURN

        return 0
