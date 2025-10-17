/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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

#include "sys_defs.h"
#include <ctype.h>

/* Utility library. */

#include "stringops.h"

int util_utf8_enable = 0;

char   *printable(char *string, int replacement)
{
    unsigned char *cp;
    int     ch;

    /*
     * XXX Replace invalid UTF8 sequences (too short, over-long encodings,
     * out-of-range code points, etc). See valid_utf8_string.c.
     */
    cp = (unsigned char *) string;
    while ((ch = *cp) != 0) {
	if (ISASCII(ch) && ISPRINT(ch)) {
	    /* ok */
	} else if (util_utf8_enable && ch >= 194 && ch <= 254
		   && cp[1] >= 128 && cp[1] < 192) {
	    /* UTF8; skip the rest of the bytes in the character. */
	    while (cp[1] >= 128 && cp[1] < 192)
		cp++;
	} else {
	    /* Not ASCII and not UTF8. */
	    *cp = replacement;
	}
	cp++;
    }
    return (string);
}
