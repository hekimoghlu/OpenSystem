/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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

#include "msg.h"
#include "vstring.h"
#include "hex_quote.h"

/* Application-specific. */

#define STR(x)	vstring_str(x)
#define LEN(x)	VSTRING_LEN(x)

/* hex_quote - raw data to quoted */

VSTRING *hex_quote(VSTRING *hex, const char *raw)
{
    const char *cp;
    int     ch;

    VSTRING_RESET(hex);
    for (cp = raw; (ch = *(unsigned const char *) cp) != 0; cp++) {
	if (ch != '%' && !ISSPACE(ch) && ISPRINT(ch)) {
	    VSTRING_ADDCH(hex, ch);
	} else {
	    vstring_sprintf_append(hex, "%%%02X", ch);
	}
    }
    VSTRING_TERMINATE(hex);
    return (hex);
}

/* hex_unquote - quoted data to raw */

VSTRING *hex_unquote(VSTRING *raw, const char *hex)
{
    const char *cp;
    int     ch;

    VSTRING_RESET(raw);
    for (cp = hex; (ch = *cp) != 0; cp++) {
	if (ch == '%') {
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
	VSTRING_ADDCH(raw, ch);
    }
    VSTRING_TERMINATE(raw);
    return (raw);
}

#ifdef TEST

 /*
  * Proof-of-concept test program: convert to hex and back.
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
    VSTRING *raw = vstring_alloc(BUFLEN);
    VSTRING *hex = vstring_alloc(100);
    ssize_t len;

    while ((len = read_buf(VSTREAM_IN, raw)) > 0) {
	hex_quote(hex, STR(raw));
	if (hex_unquote(raw, STR(hex)) == 0)
	    msg_fatal("bad input: %.100s", STR(hex));
	if (LEN(raw) != len)
	    msg_fatal("len %ld != raw len %ld", (long) len, (long) LEN(raw));
	if (vstream_fwrite(VSTREAM_OUT, STR(raw), LEN(raw)) != LEN(raw))
	    msg_fatal("write error: %m");
    }
    vstream_fflush(VSTREAM_OUT);
    vstring_free(raw);
    vstring_free(hex);
    return (0);
}

#endif
