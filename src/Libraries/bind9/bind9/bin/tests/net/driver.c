/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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
/* $Id: driver.c,v 1.11 2007/06/19 23:47:00 tbox Exp $ */

#include <config.h>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <isc/print.h>
#include <isc/string.h>
#include <isc/util.h>

#include "driver.h"

#include "testsuite.h"

#define NTESTS (sizeof(tests) / sizeof(test_t))

const char *gettime(void);
const char *test_result_totext(test_result_t);

/*
 * Not thread safe.
 */
const char *
gettime(void) {
	static char now[512];
	time_t t;

	(void)time(&t);

	strftime(now, sizeof(now) - 1,
		 "%A %d %B %H:%M:%S %Y",
		 localtime(&t));

	return (now);
}

const char *
test_result_totext(test_result_t result) {
	const char *s;
	switch (result) {
	case PASSED:
		s = "PASS";
		break;
	case FAILED:
		s = "FAIL";
		break;
	case UNTESTED:
		s = "UNTESTED";
		break;
	case UNKNOWN:
	default:
		s = "UNKNOWN";
		break;
	}

	return (s);
}

int
main(int argc, char **argv) {
	test_t *test;
	test_result_t result;
	unsigned int n_failed;
	unsigned int testno;

	UNUSED(argc);
	UNUSED(argv);

	printf("S:%s:%s\n", SUITENAME, gettime());

	n_failed = 0;
	for (testno = 0; testno < NTESTS; testno++) {
		test = &tests[testno];
		printf("T:%s:%u:A\n", test->tag, testno + 1);
		printf("A:%s\n", test->description);
		result = test->func();
		printf("R:%s\n", test_result_totext(result));
		if (result != PASSED)
			n_failed++;
	}

	printf("E:%s:%s\n", SUITENAME, gettime());

	if (n_failed > 0)
		exit(1);

	return (0);
}

