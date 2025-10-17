/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 4, 2025.
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
#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sudo_compat.h"
#include "sudo_util.h"

sudo_dso_public int main(int argc, char *argv[]);

/*
 * Test that getprogname() returns the expected result.
 * On some systems (AIX), we may have issues with symbolic links.
 */

int
main(int argc, char *argv[])
{
    const char *progbase = "progname_test";
    int ch;

    if (argc > 0)
	progbase = sudo_basename(argv[0]);
    initprogname(progbase);

    while ((ch = getopt(argc, argv, "v")) != -1) {
	switch (ch) {
	case 'v':
	    /* ignore */
	    break;
	default:
	    fprintf(stderr, "usage: %s [-v]\n", progbase);
	    return EXIT_FAILURE;
	}
    }
    argc -= optind;
    argv += optind;

    /* Make sure getprogname() matches basename of argv[0]. */
    if (strcmp(getprogname(), progbase) != 0) {
	printf("%s: FAIL: incorrect program name \"%s\"\n",
	    progbase, getprogname());
	return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
