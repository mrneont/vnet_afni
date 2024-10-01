Description of scripts in this directory:

do_00_zeropad.tcsh : prepare mask+orig dset pairs by trimming them
                     (via negative zeropadding) to the desired matrix
                     size. 
                     This assumes that the user already has
                     train|validation|test directories and mask|orig
                     subdirectories already set up appropriately.
