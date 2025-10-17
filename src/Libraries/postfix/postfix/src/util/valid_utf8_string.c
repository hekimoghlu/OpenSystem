/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 6, 2023.
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

/* Utility library. */

#include <stringops.h>

/* valid_utf8_string - validate string according to RFC 3629 */

int     valid_utf8_string(const char *str, ssize_t len)
{
    const unsigned char *end = (const unsigned char *) str + len;
    const unsigned char *cp;
    unsigned char c0, ch;

    if (len < 0)
	return (0);
    if (len <= 0)
	return (1);

    /*
     * Optimized for correct input, time, space, and for CPUs that have a
     * decent number of registers.
     */
    for (cp = (const unsigned char *) str; cp < end; cp++) {
	/* Single-byte encodings. */
	if (EXPECTED((c0 = *cp) <= 0x7f) /* we know that c0 >= 0x0 */ ) {
	     /* void */ ;
	}
	/* Two-byte encodings. */
	else if (EXPECTED(c0 <= 0xdf) /* we know that c0 >= 0x80 */ ) {
	    /* Exclude over-long encodings. */
	    if (UNEXPECTED(c0 < 0xc2)
		|| UNEXPECTED(cp + 1 >= end)
	    /* Require UTF-8 tail byte. */
		|| UNEXPECTED(((ch = *++cp) & 0xc0) != 0x80))
		return (0);
	}
	/* Three-byte encodings. */
	else if (EXPECTED(c0 <= 0xef) /* we know that c0 >= 0xe0 */ ) {
	    if (UNEXPECTED(cp + 2 >= end)
	    /* Exclude over-long encodings. */
		|| UNEXPECTED((ch = *++cp) < (c0 == 0xe0 ? 0xa0 : 0x80))
	    /* Exclude U+D800..U+DFFF. */
		|| UNEXPECTED(ch > (c0 == 0xed ? 0x9f : 0xbf))
	    /* Require UTF-8 tail byte. */
		|| UNEXPECTED(((ch = *++cp) & 0xc0) != 0x80))
		return (0);
	}
	/* Four-byte encodings. */
	else if (EXPECTED(c0 <= 0xf4) /* we know that c0 >= 0xf0 */ ) {
	    if (UNEXPECTED(cp + 3 >= end)
	    /* Exclude over-long encodings. */
		|| UNEXPECTED((ch = *++cp) < (c0 == 0xf0 ? 0x90 : 0x80))
	    /* Exclude code points above U+10FFFF. */
		|| UNEXPECTED(ch > (c0 == 0xf4 ? 0x8f : 0xbf))
	    /* Require UTF-8 tail byte. */
		|| UNEXPECTED(((ch = *++cp) & 0xc0) != 0x80)
	    /* Require UTF-8 tail byte. */
		|| UNEXPECTED(((ch = *++cp) & 0xc0) != 0x80))
		return (0);
	}
	/* Invalid: c0 >= 0xf5 */
	else {
	    return (0);
	}
    }
    return (1);
}

 /*
  * Stand-alone test program. Each string is a line without line terminator.
  */
#ifdef TEST
#include <stdlib.h>
#include <vstream.h>
#include <vstring.h>
#include <vstring_vstream.h>

#define STR(x) vstring_str(x)
#define LEN(x) VSTRING_LEN(x)

int     main(void)
{
    VSTRING *buf = vstring_alloc(1);

    while (vstring_get_nonl(buf, VSTREAM_IN) != VSTREAM_EOF) {
	vstream_printf("%c", (LEN(buf) && !valid_utf8_string(STR(buf), LEN(buf))) ?
		       '!' : ' ');
	vstream_fwrite(VSTREAM_OUT, STR(buf), LEN(buf));
	vstream_printf("\n");
    }
    vstream_fflush(VSTREAM_OUT);
    vstring_free(buf);
    exit(0);
}

#endif
