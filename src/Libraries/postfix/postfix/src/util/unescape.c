/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 22, 2022.
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
#include <ctype.h>

/* Utility library. */

#include <vstring.h>
#include <stringops.h>

/* unescape - process escape sequences */

VSTRING *unescape(VSTRING *result, const char *data)
{
    int     ch;
    int     oval;
    int     i;

#define UCHAR(cp)	((unsigned char *) (cp))
#define ISOCTAL(ch)	(ISDIGIT(ch) && (ch) != '8' && (ch) != '9')

    VSTRING_RESET(result);

    while ((ch = *UCHAR(data++)) != 0) {
	if (ch == '\\') {
	    if ((ch = *UCHAR(data++)) == 0)
		break;
	    switch (ch) {
	    case 'a':				/* \a -> audible bell */
		ch = '\a';
		break;
	    case 'b':				/* \b -> backspace */
		ch = '\b';
		break;
	    case 'f':				/* \f -> formfeed */
		ch = '\f';
		break;
	    case 'n':				/* \n -> newline */
		ch = '\n';
		break;
	    case 'r':				/* \r -> carriagereturn */
		ch = '\r';
		break;
	    case 't':				/* \t -> horizontal tab */
		ch = '\t';
		break;
	    case 'v':				/* \v -> vertical tab */
		ch = '\v';
		break;
	    case '0':				/* \nnn -> ASCII value */
	    case '1':
	    case '2':
	    case '3':
	    case '4':
	    case '5':
	    case '6':
	    case '7':
		for (oval = ch - '0', i = 0;
		     i < 2 && (ch = *UCHAR(data)) != 0 && ISOCTAL(ch);
		     i++, data++) {
		    oval = (oval << 3) | (ch - '0');
		}
		ch = oval;
		break;
	    default:				/* \any -> any */
		break;
	    }
	}
	VSTRING_ADDCH(result, ch);
    }
    VSTRING_TERMINATE(result);
    return (result);
}

/* escape - reverse transformation */

VSTRING *escape(VSTRING *result, const char *data, ssize_t len)
{
    int     ch;

    VSTRING_RESET(result);
    while (len-- > 0) {
	ch = *UCHAR(data++);
	if (ISASCII(ch)) {
	    if (ISPRINT(ch)) {
		if (ch == '\\')
		    VSTRING_ADDCH(result, ch);
		VSTRING_ADDCH(result, ch);
		continue;
	    } else if (ch == '\a') {		/* \a -> audible bell */
		vstring_strcat(result, "\\a");
		continue;
	    } else if (ch == '\b') {		/* \b -> backspace */
		vstring_strcat(result, "\\b");
		continue;
	    } else if (ch == '\f') {		/* \f -> formfeed */
		vstring_strcat(result, "\\f");
		continue;
	    } else if (ch == '\n') {		/* \n -> newline */
		vstring_strcat(result, "\\n");
		continue;
	    } else if (ch == '\r') {		/* \r -> carriagereturn */
		vstring_strcat(result, "\\r");
		continue;
	    } else if (ch == '\t') {		/* \t -> horizontal tab */
		vstring_strcat(result, "\\t");
		continue;
	    } else if (ch == '\v') {		/* \v -> vertical tab */
		vstring_strcat(result, "\\v");
		continue;
	    }
	}
	vstring_sprintf_append(result, "\\%03o", ch);
    }
    VSTRING_TERMINATE(result);
    return (result);
}

#ifdef TEST

#include <stdlib.h>
#include <string.h>
#include <msg.h>
#include <vstring_vstream.h>

int     main(int argc, char **argv)
{
    VSTRING *in = vstring_alloc(10);
    VSTRING *out = vstring_alloc(10);
    int     un_escape = 1;

    if (argc > 2 || (argc > 1 && (un_escape = strcmp(argv[1], "-e"))) != 0)
	msg_fatal("usage: %s [-e (escape)]", argv[0]);

    if (un_escape) {
	while (vstring_fgets_nonl(in, VSTREAM_IN)) {
	    unescape(out, vstring_str(in));
	    vstream_fwrite(VSTREAM_OUT, vstring_str(out), VSTRING_LEN(out));
	    VSTREAM_PUTC('\n', VSTREAM_OUT);
	}
    } else {
	while (vstring_fgets_nonl(in, VSTREAM_IN)) {
	    escape(out, vstring_str(in), VSTRING_LEN(in));
	    vstream_fwrite(VSTREAM_OUT, vstring_str(out), VSTRING_LEN(out));
	    VSTREAM_PUTC('\n', VSTREAM_OUT);
	}
    }
    vstream_fflush(VSTREAM_OUT);
    exit(0);
}

#endif
