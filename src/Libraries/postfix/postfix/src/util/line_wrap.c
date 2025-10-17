/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 21, 2025.
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
/* System library. */

#include <sys_defs.h>
#include <string.h>
#include <ctype.h>

/* Utility library. */

#include <line_wrap.h>

/* line_wrap - wrap long lines upon output */

void    line_wrap(const char *str, int len, int indent, LINE_WRAP_FN output_fn,
		          void *context)
{
    const char *start_line;
    const char *word;
    const char *next_word;
    const char *next_space;
    int     line_len;
    int     curr_len;
    int     curr_indent;

    if (indent < 0) {
	curr_indent = -indent;
	curr_len = len + indent;
    } else {
	curr_indent = 0;
	curr_len = len;
    }

    /*
     * At strategic positions, output what we have seen, after stripping off
     * trailing blanks.
     */
    for (start_line = word = str; word != 0; word = next_word) {
	next_space = word + strcspn(word, " \t");
	if (word > start_line) {
	    if (next_space - start_line > curr_len) {
		line_len = word - start_line;
		while (line_len > 0 && ISSPACE(start_line[line_len - 1]))
		    line_len--;
		output_fn(start_line, line_len, curr_indent, context);
		while (*word && ISSPACE(*word))
		    word++;
		if (start_line == str) {
		    curr_indent += indent;
		    curr_len -= indent;
		}
		start_line = word;
	    }
	}
	next_word = *next_space ? next_space + 1 : 0;
    }
    line_len = strlen(start_line);
    while (line_len > 0 && ISSPACE(start_line[line_len - 1]))
	line_len--;
    output_fn(start_line, line_len, curr_indent, context);
}
