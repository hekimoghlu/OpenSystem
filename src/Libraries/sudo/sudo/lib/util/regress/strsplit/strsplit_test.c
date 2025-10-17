/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 11, 2021.
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
#include "sudo_fatal.h"
#include "sudo_util.h"

sudo_dso_public int main(int argc, char *argv[]);

/*
 * Test that sudo_strsplit() works as expected.
 */

struct strsplit_test {
    const char *input;
    size_t input_len;
    const char **output;
};

static const char test1_in[] = " vi ";
static const char *test1_out[] = { "vi", NULL };
static const char test2_in[] = "vi -r ";
static const char *test2_out[] = { "vi", "-r", NULL };
static const char test3_in[] = "vi -r  -R abc\tdef ";
static const char *test3_out[] = { "vi", "-r", "-R", "abc", "def", NULL };
static const char test4_in[] = "vi -r  -R abc\tdef ";
static const char *test4_out[] = { "vi", "-r", "-R", "abc", NULL };
static const char test5_in[] = "";
static const char *test5_out[] = { NULL };

static struct strsplit_test test_data[] = {
    { test1_in, sizeof(test1_in) - 1, test1_out },
    { test2_in, sizeof(test2_in) - 1, test2_out },
    { test3_in, sizeof(test3_in) - 1, test3_out },
    { test4_in, sizeof(test4_in) - 5, test4_out },
    { test5_in, sizeof(test5_in) - 1, test5_out },
    { NULL, 0, NULL }
};

int
main(int argc, char *argv[])
{
    const char *cp, *ep, *input_end;
    int ch, i, j, errors = 0, ntests = 0;
    size_t len;

    initprogname(argc > 0 ? argv[0] : "strsplit_test");

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

    for (i = 0; test_data[i].input != NULL; i++) {
	input_end = test_data[i].input + test_data[i].input_len;
	cp = sudo_strsplit(test_data[i].input, input_end, " \t", &ep);
	for (j = 0; test_data[i].output[j] != NULL; j++) {
	    ntests++;
	    len = strlen(test_data[i].output[j]);
	    if ((size_t)(ep - cp) != len) {
		sudo_warnx_nodebug("failed test #%d: bad length, expected "
		    "%zu, got %zu", ntests, len, (size_t)(ep - cp));
		errors++;
		continue;
	    }
	    ntests++;
	    if (strncmp(cp, test_data[i].output[j], len) != 0) {
		sudo_warnx_nodebug("failed test #%d: expected %s, got %.*s",
		    ntests, test_data[i].output[j], (int)(ep - cp), cp);
		errors++;
		continue;
	    }
	    cp = sudo_strsplit(NULL, input_end, " \t", &ep);
	}
	ntests++;
	if (cp != NULL) {
	    sudo_warnx_nodebug("failed test #%d: extra tokens \"%.*s\"",
		ntests, (int)(input_end - cp), cp);
	    errors++;
	}
    }
    if (ntests != 0) {
	printf("%s: %d tests run, %d errors, %d%% success rate\n",
	    getprogname(), ntests, errors, (ntests - errors) * 100 / ntests);
    }
    exit(errors);
}
