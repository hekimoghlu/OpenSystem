/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif
#include <unistd.h>

#include "sudo_compat.h"
#include "sudo_util.h"
#include "sudo_fatal.h"

sudo_dso_public int main(int argc, char *argv[]);

/* sudo_strtobool() tests */
static struct strtobool_data {
    const char *bool_str;
    int value;
} strtobool_data[] = {
    { "true", true },
    { "false", false },
    { "TrUe", true },
    { "fAlSe", false },
    { "1", true },
    { "0", false },
    { "on", true },
    { "off", false },
    { "yes", true },
    { "no", false },
    { "nope", -1 },
    { "10", -1 },
    { "one", -1 },
    { "zero", -1 },
    { NULL, 0 }
};

/*
 * Simple tests for sudo_strtobool()
 */
int
main(int argc, char *argv[])
{
    struct strtobool_data *d;
    int errors = 0, ntests = 0;
    int ch, value;

    initprogname(argc > 0 ? argv[0] : "strtobool_test");

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

    for (d = strtobool_data; d->bool_str != NULL; d++) {
	ntests++;
	value = sudo_strtobool(d->bool_str);
	if (value != d->value) {
	    sudo_warnx_nodebug("FAIL: %s != %d", d->bool_str, d->value);
	    errors++;
	}
    }

    if (ntests != 0) {
	printf("%s: %d tests run, %d errors, %d%% success rate\n",
	    getprogname(), ntests, errors, (ntests - errors) * 100 / ntests);
    }

    return errors;
}
