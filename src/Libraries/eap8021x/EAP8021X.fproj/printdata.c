/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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
/* 
 * Modification History
 *
 * November 8, 2001	Dieter Siegmund
 * - created
 */

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>
#include "printdata.h"
#include "myCFUtil.h"
#include <SystemConfiguration/SCPrivate.h>

void
print_bytes_cfstr(CFMutableStringRef str, const uint8_t * data_p,
		  int n_bytes)
{
    int 		i;

    for (i = 0; i < n_bytes; i++) {
	char * space;

	if (i == 0) {
	    space = "";
	}
	else if ((i % 8) == 0) {
	    space = "  ";
	}
	else {
	    space = " ";
	}
	STRING_APPEND(str, "%s%02x", space, data_p[i]);
    }
    return;
}

void
print_data_cfstr(CFMutableStringRef str, const uint8_t * data_p,
		 int n_bytes)
{
#define CHARS_PER_LINE 	16
    char		line_buf[CHARS_PER_LINE + 1];
    int			line_pos;
    int			offset;

    for (line_pos = 0, offset = 0; offset < n_bytes; offset++, data_p++) {
	if (line_pos == 0)
	    STRING_APPEND(str, "%04x ", offset);

	line_buf[line_pos] = isprint(*data_p) ? *data_p : '.';
	STRING_APPEND(str, " %02x", *data_p);
	line_pos++;
	if (line_pos == CHARS_PER_LINE) {
	    line_buf[CHARS_PER_LINE] = '\0';
	    STRING_APPEND(str, "  %s\n", line_buf);
	    line_pos = 0;
	}
	else if (line_pos == (CHARS_PER_LINE / 2))
	    STRING_APPEND(str, " ");
    }
    if (line_pos) { /* need to finish up the line */
	char * extra_space = "";
	if (line_pos < (CHARS_PER_LINE / 2)) {
	    extra_space = " ";
	}
	for (; line_pos < CHARS_PER_LINE; line_pos++) {
	    STRING_APPEND(str, "   ");
	    line_buf[line_pos] = ' ';
	}
	line_buf[CHARS_PER_LINE] = '\0';
	STRING_APPEND(str, "  %s%s\n", extra_space, line_buf);
    }
    return;
}

void
fprint_bytes(FILE * out_f, const uint8_t * data_p, int n_bytes)
{
    CFMutableStringRef	str;

    str = CFStringCreateMutable(NULL, 0);
    if (out_f == NULL) {
	out_f = stdout;
    }
    print_bytes_cfstr(str, data_p, n_bytes);
    SCPrint(TRUE, out_f, CFSTR("%@"), str);
    CFRelease(str);
    fflush(out_f);
    return;
}

void
fprint_data(FILE * out_f, const uint8_t * data_p, int n_bytes)
{
    CFMutableStringRef	str;

    str = CFStringCreateMutable(NULL, 0);
    print_data_cfstr(str, data_p, n_bytes);
    if (out_f == NULL) {
	out_f = stdout;
    }
    SCPrint(TRUE, out_f, CFSTR("%@"), str);
    CFRelease(str);
    fflush(out_f);
    return;
}

void
print_bytes(const uint8_t * data, int len)
{
    fprint_bytes(NULL, data, len);
}

void
print_data(const uint8_t * data, int len)
{
    fprint_data(NULL, data, len);
}
