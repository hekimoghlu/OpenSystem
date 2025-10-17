/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
/*
 * Copyright 2004 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

/*
 * This is a test program designed to catch mismerges and mistranslations from
 * stabs to CTF.
 *
 * Given a file with stabs data and a file with CTF data, determine whether
 * or not all of the data structures and objects described by the stabs data
 * are present in the CTF data.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "ctftools.h"

#if defined(__APPLE__)
extern const
#endif /* __APPLE__ */ 
char *progname;
int debug_level = DEBUG_LEVEL;

static void
usage(void)
{
	fprintf(stderr, "Usage: %s ctf_file stab_file\n", progname);
}

int
main(int argc, char **argv)
{
	tdata_t *ctftd, *stabrtd, *stabtd, *difftd;
	char *ctfname, *stabname;
	int new;

	progname = argv[0];

	if (argc != 3) {
		usage();
		exit(2);
	}

	ctfname = argv[1];
	stabname = argv[2];

	stabrtd = tdata_new();
	stabtd = tdata_new();
	difftd = tdata_new();

	if (read_stabs(stabrtd, stabname, 0) != 0)
		merge_into_master(NULL, stabrtd, stabtd, NULL, 1);
	else if (read_ctf(&stabname, 1, NULL, read_ctf_save_cb, &stabtd, 0)
	    == 0)
		terminate("%s doesn't have stabs or CTF\n", stabname);

	if (read_ctf(&ctfname, 1, NULL, read_ctf_save_cb, &ctftd, 0) == 0)
		terminate("%s doesn't contain CTF data\n", ctfname);

	merge_into_master(NULL, stabtd, ctftd, difftd, 0);

	if ((new = hash_count(difftd->td_iihash)) != 0) {
		(void) hash_iter(difftd->td_iihash, (int (*)())iidesc_dump,
		    NULL);
		terminate("%s grew by %d\n", stabname, new);
	}

	return (0);
}
