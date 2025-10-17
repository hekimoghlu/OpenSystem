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
#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sudoers.h"

sudo_dso_public int main(int argc, char *argv[]);

int
main(int argc, char *argv[])
{
    FILE *fp = stdin;
    char pattern[1024], string[1024];
    int errors = 0, tests = 0, got, want;

    initprogname(argc > 0 ? argv[0] : "check_env_pattern");

    if (argc > 1) {
	if ((fp = fopen(argv[1], "r")) == NULL) {
	    perror(argv[1]);
	    exit(EXIT_FAILURE);
	}
    }

    /*
     * Read in test file, which is formatted thusly:
     *
     * pattern string 1/0
     *
     */
    for (;;) {
	bool full_match = false;

	got = fscanf(fp, "%s %s %d\n", pattern, string, &want);
	if (got == EOF)
	    break;
	if (got == 3) {
	    got = matches_env_pattern(pattern, string, &full_match);
	    if (full_match)
		got++;
	    if (got != want) {
		fprintf(stderr,
		    "%s: %s %s: want %d, got %d\n",
		    getprogname(), pattern, string, want, got);
		errors++;
	    }
	    tests++;
	}
    }
    if (tests != 0) {
	printf("%s: %d test%s run, %d errors, %d%% success rate\n",
	    getprogname(), tests, tests == 1 ? "" : "s", errors,
	    (tests - errors) * 100 / tests);
    }
    exit(errors);
}
