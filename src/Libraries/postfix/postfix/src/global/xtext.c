/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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

#include "msg.h"
#include "vstring.h"
#include "xtext.h"

/* Application-specific. */

#define STR(x)	vstring_str(x)
#define LEN(x)	VSTRING_LEN(x)

/* xtext_quote_append - append unquoted data to quoted data */

VSTRING *xtext_quote_append(VSTRING *quoted, const char *unquoted,
			            const char *special)
{
    const char *cp;
    int     ch;

    for (cp = unquoted; (ch = *(unsigned const char *) cp) != 0; cp++) {
	if (ch != '+' && ch > 32 && ch < 127
	    && (*special == 0 || strchr(special, ch) == 0)) {
	    VSTRING_ADDCH(quoted, ch);
	} else {
	    vstring_sprintf_append(quoted, "+%02X", ch);
	}
    }
    VSTRING_TERMINATE(quoted);
    return (quoted);
}

/* xtext_quote - unquoted data to quoted */

VSTRING *xtext_quote(VSTRING *quoted, const char *unquoted, const char *special)
{
    VSTRING_RESET(quoted);
    xtext_quote_append(quoted, unquoted, special);
    return (quoted);
}

/* xtext_unquote_append - quoted data to unquoted */

VSTRING *xtext_unquote_append(VSTRING *unquoted, const char *quoted)
{
    const unsigned char *cp;
    int     ch;

    for (cp = (const unsigned char *) quoted; (ch = *cp) != 0; cp++) {
	if (ch == '+') {
	    if (ISDIGIT(cp[1]))
		ch = (cp[1] - '0') << 4;
	    else if (cp[1] >= 'a' && cp[1] <= 'f')
		ch = (cp[1] - 'a' + 10) << 4;
	    else if (cp[1] >= 'A' && cp[1] <= 'F')
		ch = (cp[1] - 'A' + 10) << 4;
	    else
		return (0);
	    if (ISDIGIT(cp[2]))
		ch |= (cp[2] - '0');
	    else if (cp[2] >= 'a' && cp[2] <= 'f')
		ch |= (cp[2] - 'a' + 10);
	    else if (cp[2] >= 'A' && cp[2] <= 'F')
		ch |= (cp[2] - 'A' + 10);
	    else
		return (0);
	    cp += 2;
	}
	VSTRING_ADDCH(unquoted, ch);
    }
    VSTRING_TERMINATE(unquoted);
    return (unquoted);
}
/* xtext_unquote - quoted data to unquoted */

VSTRING *xtext_unquote(VSTRING *unquoted, const char *quoted)
{
    VSTRING_RESET(unquoted);
    return (xtext_unquote_append(unquoted, quoted) ? unquoted : 0);
}

#ifdef TEST

 /*
  * Proof-of-concept test program: convert to quoted and back.
  */
#include <vstream.h>

#define BUFLEN 1024

static ssize_t read_buf(VSTREAM *fp, VSTRING *buf)
{
    ssize_t len;

    VSTRING_RESET(buf);
    len = vstream_fread(fp, STR(buf), vstring_avail(buf));
    VSTRING_AT_OFFSET(buf, len);		/* XXX */
    VSTRING_TERMINATE(buf);
    return (len);
}

int     main(int unused_argc, char **unused_argv)
{
    VSTRING *unquoted = vstring_alloc(BUFLEN);
    VSTRING *quoted = vstring_alloc(100);
    ssize_t len;

    /*
     * Negative tests.
     */
    if (xtext_unquote(unquoted, "++1") != 0)
	msg_warn("undetected error pattern 1");
    if (xtext_unquote(unquoted, "+2+") != 0)
	msg_warn("undetected error pattern 2");

    /*
     * Positive tests.
     */
    while ((len = read_buf(VSTREAM_IN, unquoted)) > 0) {
	xtext_quote(quoted, STR(unquoted), "+=");
	if (xtext_unquote(unquoted, STR(quoted)) == 0)
	    msg_fatal("bad input: %.100s", STR(quoted));
	if (LEN(unquoted) != len)
	    msg_fatal("len %ld != unquoted len %ld",
		      (long) len, (long) LEN(unquoted));
	if (vstream_fwrite(VSTREAM_OUT, STR(unquoted), LEN(unquoted)) != LEN(unquoted))
	    msg_fatal("write error: %m");
    }
    vstream_fflush(VSTREAM_OUT);
    vstring_free(unquoted);
    vstring_free(quoted);
    return (0);
}

#endif
