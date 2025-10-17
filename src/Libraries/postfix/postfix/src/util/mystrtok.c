/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 10, 2023.
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
#include <string.h>

/* Utility library. */

#include "stringops.h"

/* mystrtok - safe tokenizer */

char   *mystrtok(char **src, const char *sep)
{
    char   *start = *src;
    char   *end;

    /*
     * Skip over leading delimiters.
     */
    start += strspn(start, sep);
    if (*start == 0) {
	*src = start;
	return (0);
    }

    /*
     * Separate off one token.
     */
    end = start + strcspn(start, sep);
    if (*end != 0)
	*end++ = 0;
    *src = end;
    return (start);
}

/* mystrtokq - safe tokenizer with quoting support */

char   *mystrtokq(char **src, const char *sep, const char *parens)
{
    char   *start = *src;
    static char   *cp;
    int     ch;
    int     level;

    /*
     * Skip over leading delimiters.
     */
    start += strspn(start, sep);
    if (*start == 0) {
	*src = start;
	return (0);
    }

    /*
     * Parse out the next token.
     */
    for (level = 0, cp = start; (ch = *(unsigned char *) cp) != 0; cp++) {
	if (ch == parens[0]) {
	    level++;
        } else if (level > 0 && ch == parens[1]) {
	    level--;
	} else if (level == 0 && strchr(sep, ch) != 0) {
	    *cp++ = 0;
	    break;
	}
    }
    *src = cp;
    return (start);
}

#ifdef TEST

 /*
  * Test program: read lines from stdin, split on whitespace.
  */
#include "vstring.h"
#include "vstream.h"
#include "vstring_vstream.h"

int     main(void)
{
    VSTRING *vp = vstring_alloc(100);
    char   *start;
    char   *str;

    while (vstring_fgets(vp, VSTREAM_IN) && VSTRING_LEN(vp) > 0) {
	start = vstring_str(vp);
	if (strchr(start, CHARS_BRACE[0]) == 0) {
	    while ((str = mystrtok(&start, CHARS_SPACE)) != 0)
		vstream_printf(">%s<\n", str);
	} else {
	    while ((str = mystrtokq(&start, CHARS_SPACE, CHARS_BRACE)) != 0)
		vstream_printf(">%s<\n", str);
	}
	vstream_fflush(VSTREAM_OUT);
    }
    vstring_free(vp);
    return (0);
}

#endif
