/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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

#include <vstring.h>

/* Global library. */

#include "quote_821_local.h"

/* Application-specific. */

#define YES	1
#define	NO	0

/* is_821_dot_string - is this local-part an rfc 821 dot-string? */

static int is_821_dot_string(const char *local_part, const char *end, int flags)
{
    const char *cp;
    int     ch;

    /*
     * Detect any deviations from the definition of dot-string. We could use
     * lookup tables to speed up some of the work, but hey, how large can a
     * local-part be anyway?
     */
    if (local_part == end || local_part[0] == 0 || local_part[0] == '.')
	return (NO);
    for (cp = local_part; cp < end && (ch = *(unsigned char *) cp) != 0; cp++) {
	if (ch == '.' && cp[1] == '.')
	    return (NO);
	if (ch > 127 && !(flags & QUOTE_FLAG_8BITCLEAN))
	    return (NO);
	if (ch == ' ')
	    return (NO);
	if (ISCNTRL(ch))
	    return (NO);
	if (ch == '<' || ch == '>'
	    || ch == '(' || ch == ')'
	    || ch == '[' || ch == ']'
	    || ch == '\\' || ch == ','
	    || ch == ';' || ch == ':'
	    || (ch == '@' && !(flags & QUOTE_FLAG_EXPOSE_AT)) || ch == '"')
	    return (NO);
    }
    if (cp[-1] == '.')
	return (NO);
    return (YES);
}

/* make_821_quoted_string - make quoted-string from local-part */

static VSTRING *make_821_quoted_string(VSTRING *dst, const char *local_part,
				               const char *end, int flags)
{
    const char *cp;
    int     ch;

    /*
     * Put quotes around the result, and prepend a backslash to characters
     * that need quoting when they occur in a quoted-string.
     */
    VSTRING_ADDCH(dst, '"');
    for (cp = local_part; cp < end && (ch = *(unsigned char *) cp) != 0; cp++) {
	if ((ch > 127 && !(flags & QUOTE_FLAG_8BITCLEAN))
	    || ch == '\r' || ch == '\n' || ch == '"' || ch == '\\')
	    VSTRING_ADDCH(dst, '\\');
	VSTRING_ADDCH(dst, ch);
    }
    VSTRING_ADDCH(dst, '"');
    VSTRING_TERMINATE(dst);
    return (dst);
}

/* quote_821_local_flags - quote local part of address according to rfc 821 */

VSTRING *quote_821_local_flags(VSTRING *dst, const char *addr, int flags)
{
    const char   *at;

    /*
     * According to RFC 821, a local-part is a dot-string or a quoted-string.
     * We first see if the local-part is a dot-string. If it is not, we turn
     * it into a quoted-string. Anything else would be too painful.
     */
    if ((at = strrchr(addr, '@')) == 0)		/* just in case */
	at = addr + strlen(addr);		/* should not happen */
    if ((flags & QUOTE_FLAG_APPEND) == 0)
	VSTRING_RESET(dst);
    if (is_821_dot_string(addr, at, flags)) {
	return (vstring_strcat(dst, addr));
    } else {
	make_821_quoted_string(dst, addr, at, flags & QUOTE_FLAG_8BITCLEAN);
	return (vstring_strcat(dst, at));
    }
}

#ifdef TEST

 /*
  * Test program for local-part quoting as per rfc 821
  */
#include <stdlib.h>
#include <vstream.h>
#include <vstring_vstream.h>
#include "quote_821_local.h"

int     main(void)
{
    VSTRING *src = vstring_alloc(100);
    VSTRING *dst = vstring_alloc(100);

    while (vstring_fgets_nonl(src, VSTREAM_IN)) {
	vstream_fprintf(VSTREAM_OUT, "%s\n",
			vstring_str(quote_821_local(dst, vstring_str(src))));
	vstream_fflush(VSTREAM_OUT);
    }
    exit(0);
}

#endif
