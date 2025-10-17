/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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
#include <string.h>
#include <err.h>
#include "windlocl.h"

static const char *failing_testcases[] = {
    "\x80",
    "\xFF",
    "\xC0",
    "\xDF",
    "\xE0",
    "\xEF",
    "\xF0",
    "\xF7",
    "\xC0\x01",
    "\xC0\x7F",
    "\xC0\xFF",
    "\xC0\x80\x80",
    "\xE0\x01",
    "\xE0\x7F",
    "\xE0\x80",
    "\xE0\xFF",
    "\xE0\x80\x20",
    "\xE0\x80\xFF",
    "\xE0\x80\x80\x80",
    "\xF0\x01",
    "\xF0\x80",
    "\xF0\x80\x01",
    "\xF0\x80\x80",
    "\xF0\x80\x80\x01",
    "\xF0\x80\x80\xFF",
    NULL
};

#define MAX_LENGTH 10

struct testcase {
    const char *utf8_str;
    size_t len;
    uint32_t u[MAX_LENGTH];
    int invalid_ucs2;
};

static const struct testcase testcases[] = {
    {"", 0, {0}},
    {"\x01", 1, {1}},
    {"\x7F", 1, {0x7F}},
    {"\x01\x7F", 2, {0x01, 0x7F}},
    {"\xC0\x80", 1, {0}},
    {"\xC0\x81", 1, {1}},
    {"\xC1\x80", 1, {0x40}},
    {"\xDF\xBF", 1, {0x7FF}},
    {"\xE0\x80\x80", 1, {0}},
    {"\xE0\x80\x81", 1, {1}},
    {"\xE0\x81\x80", 1, {0x40}},
    {"\xE1\x80\x80", 1, {0x1000}},
    {"\xEF\xBF\xBF", 1, {0xFFFF}},
    {"\xF0\x80\x80\x80", 1, {0}},
    {"\xF0\x80\x80\x81", 1, {1}},
    {"\xF0\x80\x81\x80", 1, {0x40}},
    {"\xF0\x81\x80\x80", 1, {0x1000}},
    {"\xF1\x80\x80\x80", 1, {0x40000}},
    {"\xF7\xBF\xBF\xBF", 1, {0X1FFFFF}, 1},
};

int
main(void)
{
    unsigned failures = 0;
    unsigned i;
    const char **s;
    int ret;
    size_t len, len2;
    uint32_t u[MAX_LENGTH];
    char str[MAX_LENGTH * 4];

    for (s = failing_testcases; *s != NULL; ++s) {
	len = MAX_LENGTH;
	ret = wind_utf8ucs4(*s, u, &len);
	if (ret == 0) {
	    printf("utf8 decode of \"%s\" should have failed\n", *s);
	    ++failures;
	}
    }

    for (i = 0; i < sizeof(testcases)/sizeof(testcases[0]); ++i) {
	const struct testcase *t = &testcases[i];

	ret = wind_utf8ucs4_length(t->utf8_str, &len);
	if (ret) {
	    printf("utf8ucs4 length of \"%s\" should have succeeded\n",
		   t->utf8_str);
	    ++failures;
	    continue;
	}
	if (len != t->len) {
	    printf("utf8ucs4_length of \"%s\" has wrong length: "
		   "expected: %u, actual: %u\n",
		   t->utf8_str, (unsigned int)t->len, (unsigned int)len);
	    ++failures;
	    continue;
	}

	len = MAX_LENGTH;
	ret = wind_utf8ucs4(t->utf8_str, u, &len);
	if (ret) {
	    printf("utf8 decode of \"%s\" should have succeeded\n",
		   t->utf8_str);
	    ++failures;
	    continue;
	}
	if (len != t->len) {
	    printf("utf8 decode of \"%s\" has wrong length: "
		   "expected: %u, actual: %u\n",
		   t->utf8_str, (unsigned int)t->len, (unsigned int)len);
	    ++failures;
	    continue;
	}
	if (memcmp(t->u, u, len * sizeof(uint32_t)) != 0) {
	    printf("utf8 decode of \"%s\" has wrong data\n",
		   t->utf8_str);
	    ++failures;
	    continue;
	}
	if (t->invalid_ucs2 == 0) {
	    len2 = sizeof(str);
	    ret = wind_ucs4utf8(u, len, str, &len2);
	    if (ret) {
		printf("ucs4 decode of \"%s\" should have succeeded\n",
		       t->utf8_str);
		++failures;
		continue;
	    }
	}
    }

    return failures != 0;
}
