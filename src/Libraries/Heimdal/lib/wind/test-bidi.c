/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include <stdio.h>
#include "windlocl.h"

#define MAX_LENGTH 10

struct test {
    unsigned len;
    uint32_t vals[MAX_LENGTH];
};

static struct test passing_cases[] = {
    {0, {0}},
    {1, {0x0041}},
    {1, {0x05be}},
};

static struct test failing_cases[] = {
    {2, {0x05be, 0x0041}},
    {3, {0x05be, 0x0041, 0x05be}},
};

int
main(void)
{
    unsigned i;
    unsigned failures = 0;

    for (i = 0; i < sizeof(passing_cases)/sizeof(passing_cases[0]); ++i) {
	const struct test *t = &passing_cases[i];
	if (_wind_stringprep_testbidi(t->vals, t->len, WIND_PROFILE_NAME)) {
	    printf ("passing case %u failed\n", i);
	    ++failures;
	}
    }

    for (i = 0; i < sizeof(failing_cases)/sizeof(failing_cases[0]); ++i) {
	const struct test *t = &failing_cases[i];
	if (!_wind_stringprep_testbidi(t->vals, t->len, WIND_PROFILE_NAME)) {
	    printf ("failing case %u passed\n", i);
	    ++failures;
	}
    }

    return failures != 0;
}
