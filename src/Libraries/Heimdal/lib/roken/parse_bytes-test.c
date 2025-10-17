/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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

#include "roken.h"
#include "parse_bytes.h"

static struct testcase {
    int canonicalp;
    int val;
    const char *def_unit;
    const char *str;
} tests[] = {
    {0, 0, NULL, "0 bytes"},
    {1, 0, NULL, "0"},
    {0, 1, NULL, "1"},
    {1, 1, NULL, "1 byte"},
    {0, 0, "kilobyte", "0"},
    {0, 1024, "kilobyte", "1"},
    {1, 1024, "kilobyte", "1 kilobyte"},
    {1, 1024 * 1024, NULL, "1 megabyte"},
    {0, 1025, NULL, "1 kilobyte 1"},
    {1, 1025, NULL, "1 kilobyte 1 byte"},
};

int
main(int argc, char **argv)
{
    int i;
    int ret = 0;

    for (i = 0; i < sizeof(tests)/sizeof(tests[0]); ++i) {
	char buf[256];
	int val = parse_bytes (tests[i].str, tests[i].def_unit);
	int len;

	if (val != tests[i].val) {
	    printf ("parse_bytes (%s, %s) = %d != %d\n",
		    tests[i].str,
		    tests[i].def_unit ? tests[i].def_unit : "none",
		    val, tests[i].val);
	    ++ret;
	}
	if (tests[i].canonicalp) {
	    len = unparse_bytes (tests[i].val, buf, sizeof(buf));
	    if (strcmp (tests[i].str, buf) != 0) {
		printf ("unparse_bytes (%d) = \"%s\" != \"%s\"\n",
			tests[i].val, buf, tests[i].str);
		++ret;
	    }
	}
    }
    if (ret) {
	printf ("%d errors\n", ret);
	return 1;
    } else
	return 0;
}
