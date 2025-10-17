/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
#include <time.h>

#define SUDO_ERROR_WRAP 0

#include "sudo_compat.h"
#include "sudo_util.h"
#include "sudoers_debug.h"
#include "parse.h"

sudo_dso_public int main(int argc, char *argv[]);

const struct gentime_test {
    const char *gentime;
    time_t unixtime;
} tests[] = {
    { "199412161032ZZ", -1 },
    { "199412161032Z", 787573920 },
    { "199412160532-0500", 787573920 },
    { "199412160532-05000", -1 },
    { "199412160532", 787573920 },		/* local time is EST */
    { "20170214083000-0500", 1487079000 },
    { "201702140830-0500", 1487079000 },
    { "201702140830", 1487079000 },		/* local time is EST */
    { "201702140830.3-0500", 1487079018 },
    { "201702140830,3-0500", 1487079018 },
    { "20170214083000.5Z", 1487061000 },
    { "20170214083000,5Z", 1487061000 },
    { "201702142359.4Z", 1487116764 },
    { "201702142359,4Z", 1487116764 },
    { "2017021408.5Z", 1487061000 },
    { "2017021408,5Z", 1487061000 },
    { "20170214Z", -1 },
};

int
main(int argc, char *argv[])
{
    const int ntests = nitems(tests);
    int i, errors = 0;
    time_t result;

    initprogname(argc > 0 ? argv[0] : "check_gentime");

    /* Do local time tests in Eastern Standard Time. */
    putenv((char *)"TZ=EST5EST5");
    tzset();

    for (i = 0; i < ntests; i++) {
	result = parse_gentime(tests[i].gentime);
	if (result != tests[i].unixtime) {
	    fprintf(stderr, "check_gentime[%d]: %s: expected %lld, got %lld\n",
		i, tests[i].gentime,
		(long long)tests[i].unixtime, (long long)result);
	    errors++;
	}
    }
    if (ntests != 0) {
	printf("%s: %d tests run, %d errors, %d%% success rate\n",
	    getprogname(), ntests, errors, (ntests - errors) * 100 / ntests);
    }
    exit(errors);
}
