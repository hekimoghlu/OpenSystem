/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 15, 2022.
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
 * Public domain, 2008, Todd C. Miller <Todd.Miller@sudo.ws>
 */

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef HAVE_FNMATCH
# include <fnmatch.h>
#else
# include "compat/fnmatch.h"
#endif

#include "sudo_compat.h"
#include "sudo_util.h"

sudo_dso_public int main(int argc, char *argv[]);

int
main(int argc, char *argv[])
{
	FILE *fp = stdin;
	char pattern[1024], string[1024], flagstr[1024];
	int ch, errors = 0, ntests = 0, flags, got, want;

	initprogname(argc > 0 ? argv[0] : "fnm_test");

	while ((ch = getopt(argc, argv, "v")) != -1) {
	    switch (ch) {
	    case 'v':
		/* ignore */
		break;
	    default:
		fprintf(stderr, "usage: %s [-v]\n", getprogname());
		return EXIT_FAILURE;
	    }
	}
	argc -= optind;
	argv += optind;

	if (argc > 0) {
		if ((fp = fopen(argv[0], "r")) == NULL) {
			perror(argv[0]);
			return EXIT_FAILURE;
		}
	}

	/*
	 * Read in test file, which is formatted thusly:
	 *
	 * pattern string flags expected_result
	 *
	 */
	for (;;) {
		got = fscanf(fp, "%s %s %s %d\n", pattern, string, flagstr,
		    &want);
		if (got == EOF)
			break;
		if (got == 4) {
			flags = 0;
			if (strcmp(flagstr, "FNM_NOESCAPE") == 0)
				flags |= FNM_NOESCAPE;
			else if (strcmp(flagstr, "FNM_PATHNAME") == 0)
				flags |= FNM_PATHNAME;
			else if (strcmp(flagstr, "FNM_PERIOD") == 0)
				flags |= FNM_PERIOD;
			else if (strcmp(flagstr, "FNM_LEADING_DIR") == 0)
				flags |= FNM_LEADING_DIR;
			else if (strcmp(flagstr, "FNM_CASEFOLD") == 0)
				flags |= FNM_CASEFOLD;
			got = fnmatch(pattern, string, flags);
			if (got != want) {
				fprintf(stderr,
				    "fnmatch: %s %s %d: want %d, got %d\n",
				    pattern, string, flags, want, got);
				errors++;
			}
			ntests++;
		}
	}
	if (ntests != 0) {
		printf("fnmatch: %d test%s run, %d errors, %d%% success rate\n",
		    ntests, ntests == 1 ? "" : "s", errors,
		    (ntests - errors) * 100 / ntests);
	}
	return errors;
}
