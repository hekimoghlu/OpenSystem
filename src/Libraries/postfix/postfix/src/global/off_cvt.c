/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 26, 2023.
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
#include <sys/types.h>
#include <ctype.h>

/* Utility library. */

#include <msg.h>
#include <vstring.h>

/* Global library. */

#include "off_cvt.h"

/* Application-specific. */

#define STR	vstring_str
#define END	vstring_end
#define SWAP(type, a, b) { type temp; temp = a; a = b; b = temp; }

/* off_cvt_string - string to number */

off_t   off_cvt_string(const char *str)
{
    int     ch;
    off_t   result;
    off_t   digit_value;

    /*
     * Detect overflow before it happens. Code that attempts to detect
     * overflow after-the-fact makes assumptions about undefined behavior.
     * Compilers may invalidate such assumptions.
     */
    for (result = 0; (ch = *(unsigned char *) str) != 0; str++) {
	if (!ISDIGIT(ch))
	    return (-1);
	digit_value = ch - '0';
	if (result > OFF_T_MAX / 10
	    || (result *= 10) > OFF_T_MAX - digit_value)
	    return (-1);
	result += digit_value;
    }
    return (result);
}

/* off_cvt_number - number to string */

VSTRING *off_cvt_number(VSTRING *buf, off_t offset)
{
    static char digs[] = "0123456789";
    char   *start;
    char   *last;
    int     i;

    /*
     * Sanity checks
     */
    if (offset < 0)
	msg_panic("off_cvt_number: negative offset -%s",
		  STR(off_cvt_number(buf, -offset)));

    /*
     * First accumulate the result, backwards.
     */
    VSTRING_RESET(buf);
    while (offset != 0) {
	VSTRING_ADDCH(buf, digs[offset % 10]);
	offset /= 10;
    }
    VSTRING_TERMINATE(buf);

    /*
     * Then, reverse the result.
     */
    start = STR(buf);
    last = END(buf) - 1;
    for (i = 0; i < VSTRING_LEN(buf) / 2; i++)
	SWAP(int, start[i], last[-i]);
    return (buf);
}

#ifdef TEST

 /*
  * Proof-of-concept test program. Read a number from stdin, convert to
  * off_t, back to string, and print the result.
  */
#include <vstream.h>
#include <vstring_vstream.h>

int     main(int unused_argc, char **unused_argv)
{
    VSTRING *buf = vstring_alloc(100);
    off_t   offset;

    while (vstring_fgets_nonl(buf, VSTREAM_IN)) {
	if (STR(buf)[0] == '#' || STR(buf)[0] == 0)
	    continue;
	if ((offset = off_cvt_string(STR(buf))) < 0) {
	    msg_warn("bad input %s", STR(buf));
	} else {
	    vstream_printf("%s\n", STR(off_cvt_number(buf, offset)));
	}
	vstream_fflush(VSTREAM_OUT);
    }
    vstring_free(buf);
    return (0);
}

#endif
