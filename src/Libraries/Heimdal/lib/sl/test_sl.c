/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 5, 2024.
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
#include "sl_locl.h"

struct {
    int ok;
    const char *line;
    int argc;
    const char *argv[4];
} lines[] = {
    { 1, "", 1, { "" } },
    { 1, "foo", 1, { "foo" } },
    { 1, "foo bar", 2, { "foo", "bar" }},
    { 1, "foo bar baz", 3, { "foo", "bar", "baz" }},
    { 1, "foobar baz", 2, { "foobar", "baz" }},
    { 1, " foo", 1, { "foo" } },
    { 1, "foo   ", 1, { "foo" } },
    { 1, " foo  ", 1, { "foo" } },
    { 1, " foo  bar", 2, { "foo", "bar" } },
    { 1, "foo\\ bar", 1, { "foo bar" } },
    { 1, "\"foo bar\"", 1, { "foo bar" } },
    { 1, "\"foo\\ bar\"", 1, { "foo bar" } },
    { 1, "\"foo\\\" bar\"", 1, { "foo\" bar" } },
    { 1, "\"\"f\"\"oo\"\"", 1, { "foo" } },
    { 1, "\"foobar\"baz", 1, { "foobarbaz" }},
    { 1, "foo\tbar baz", 3, { "foo", "bar", "baz" }},
    { 1, "\"foo bar\" baz", 2, { "foo bar", "baz" }},
    { 1, "\"foo bar baz\"", 1, { "foo bar baz" }},
    { 1, "\\\"foo bar baz", 3, { "\"foo", "bar", "baz" }},
    { 1, "\\ foo bar baz", 3, { " foo", "bar", "baz" }},
    { 0, "\\", 0, { "" }},
    { 0, "\"", 0, { "" }}
};

int
main(int argc, char **argv)
{
    int ret, i;

    for (i = 0; i < sizeof(lines)/sizeof(lines[0]); i++) {
	int j, rargc = 0;
	char **rargv = NULL;
	char *buf = strdup(lines[i].line);

	ret = sl_make_argv(buf, &rargc, &rargv);
	if (ret) {
	    if (!lines[i].ok)
		goto next;
	    errx(1, "sl_make_argv test %d failed", i);
	} else if (!lines[i].ok)
	    errx(1, "sl_make_argv passed test %d when it shouldn't", i);
	if (rargc != lines[i].argc)
	    errx(1, "result argc (%d) != should be argc (%d) for test %d",
		 rargc, lines[i].argc, i);
	for (j = 0; j < rargc; j++)
	    if (strcmp(rargv[j], lines[i].argv[j]) != 0)
		errx(1, "result argv (%s) != should be argv (%s) for test %d",
		     rargv[j], lines[i].argv[j], i);
    next:
	free(buf);
	free(rargv);
    }

    return 0;
}
