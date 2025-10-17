/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
#include <unistd.h>

#include "sudo_compat.h"
#include "sudo_util.h"
#include "sudo_fatal.h"

sudo_dso_public int main(int argc, char *argv[]);

/* sudo_strtomode() tests */
static struct strtomode_data {
    const char *mode_str;
    mode_t mode;
} strtomode_data[] = {
    { "755", 0755 },
    { "007", 007 },
    { "7", 7 },
    { "8", (mode_t)-1 },
    { NULL, 0 }
};

/*
 * Simple tests for sudo_strtomode().
 */
int
main(int argc, char *argv[])
{
    struct strtomode_data *d;
    const char *errstr;
    int ch, errors = 0, ntests = 0;
    mode_t mode;

    initprogname(argc > 0 ? argv[0] : "strtomode_test");

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

    for (d = strtomode_data; d->mode_str != NULL; d++) {
	ntests++;
	errstr = "some error";
	mode = sudo_strtomode(d->mode_str, &errstr);
	if (errstr != NULL) {
	    if (d->mode != (mode_t)-1) {
		sudo_warnx_nodebug("FAIL: %s: %s", d->mode_str, errstr);
		errors++;
	    }
	} else if (mode != d->mode) {
	    sudo_warnx_nodebug("FAIL: %s != 0%o", d->mode_str,
		(unsigned int) d->mode);
	    errors++;
	}
    }

    if (ntests != 0) {
	printf("%s: %d tests run, %d errors, %d%% success rate\n",
	    getprogname(), ntests, errors, (ntests - errors) * 100 / ntests);
    }

    return errors;
}
