/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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
#include <limits.h>
#include <errno.h>
#include <unistd.h>

#include "sudo_compat.h"
#include "sudo_util.h"
#include "sudo_fatal.h"

sudo_dso_public int main(int argc, char *argv[]);

/* sudo_strtonum() tests */
static struct strtonum_data {
    const char *str;
    long long minval;
    long long maxval;
    long long retval;
    int errnum;
} strtonum_data[] = {
    { "0,1", LLONG_MIN, LLONG_MAX, 0, EINVAL },
    { "0", INT_MAX, INT_MIN, 0, EINVAL },
    { "", 0, UINT_MAX, 0, EINVAL },
    { " ", 0, UINT_MAX, 0, EINVAL },
    { "-1 ", 0, UINT_MAX, 0, EINVAL },
    { "9223372036854775808X", LLONG_MIN, LLONG_MAX, 0, EINVAL },
    { "-9223372036854775809X", LLONG_MIN, LLONG_MAX, 0, EINVAL },

    { "10", 0, 255, 10, 0 },
    { "-1", 0, UINT_MAX, 0, ERANGE },

    { "-40", -100, -50, 0, ERANGE },
    { "-60", -100, -50, -60, 0 },
    { "-200", -100, -50, 0, ERANGE },

    { "42", 42, 42, 42, 0 },
    { "-42", -42, -42, -42, 0 },

    { "4294967295", 0, UINT_MAX, UINT_MAX, 0 },
    { "4294967295", INT_MIN, INT_MAX, 0, ERANGE },
    { "4294967296", 0, UINT_MAX, 0, ERANGE },

    { "2147483647", INT_MIN, INT_MAX, INT_MAX, 0 },
    { "-2147483648", INT_MIN, INT_MAX, INT_MIN, 0 },
    { "2147483648", INT_MIN, INT_MAX, 0, ERANGE },
    { "-2147483649", INT_MIN, INT_MAX, 0, ERANGE },

    { "9223372036854775807", LLONG_MIN, LLONG_MAX, LLONG_MAX, 0 },
    { "-9223372036854775808", LLONG_MIN, LLONG_MAX, LLONG_MIN, 0 },
    { "9223372036854775808", LLONG_MIN, LLONG_MAX, 0, ERANGE },
    { "-9223372036854775809", LLONG_MIN, LLONG_MAX, 0, ERANGE },

    { NULL, 0, 0, 0, 0 }
};

/*
 * Simple tests for sudo_strtonum()
 */
int
main(int argc, char *argv[])
{
    int ch, errors = 0, ntests = 0;
    struct strtonum_data *d;
    const char *errstr;
    long long value;

    initprogname(argc > 0 ? argv[0] : "strtonum_test");

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

    for (d = strtonum_data; d->str != NULL; d++) {
	ntests++;
	errstr = "some error";
	value = sudo_strtonum(d->str, d->minval, d->maxval, &errstr);
	if (d->errnum != 0) {
	    if (errstr == NULL) {
		sudo_warnx_nodebug("FAIL: \"%s\": missing errstr for errno %d",
		    d->str, d->errnum);
		errors++;
	    } else if (value != 0) {
		sudo_warnx_nodebug("FAIL: %s should return 0 on error",
		    d->str);
		errors++;
	    } else if (errno != d->errnum) {
		sudo_warnx_nodebug("FAIL: \"%s\": errno mismatch, %d != %d",
		    d->str, errno, d->errnum);
		errors++;
	    }
	} else if (errstr != NULL) {
	    sudo_warnx_nodebug("FAIL: \"%s\": %s", d->str, errstr);
	    errors++;
	} else if (value != d->retval) {
	    sudo_warnx_nodebug("FAIL: %s != %lld", d->str, d->retval);
	    errors++;
	}
    }

    if (ntests != 0) {
	printf("%s: %d tests run, %d errors, %d%% success rate\n",
	    getprogname(), ntests, errors, (ntests - errors) * 100 / ntests);
    }

    return errors;
}
