/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 7, 2023.
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

#include "msg.h"
#include "vstream.h"
#include "vstring.h"
#include "readlline.h"

#define STR(x) vstring_str(x)
#define LEN(x) VSTRING_LEN(x)
#define END(x) vstring_end(x)

/* readllines - read one logical line */

VSTRING *readllines(VSTRING *buf, VSTREAM *fp, int *lineno, int *first_line)
{
    int     ch;
    int     next;
    ssize_t start;
    char   *cp;

    VSTRING_RESET(buf);

    /*
     * Ignore comment lines, all whitespace lines, and empty lines. Terminate
     * at EOF or at the beginning of the next logical line.
     */
    for (;;) {
	/* Read one line, possibly not newline terminated. */
	start = LEN(buf);
	while ((ch = VSTREAM_GETC(fp)) != VSTREAM_EOF && ch != '\n')
	    VSTRING_ADDCH(buf, ch);
	if (lineno != 0 && (ch == '\n' || LEN(buf) > start))
	    *lineno += 1;
	/* Ignore comment line, all whitespace line, or empty line. */
	for (cp = STR(buf) + start; cp < END(buf) && ISSPACE(*cp); cp++)
	     /* void */ ;
	if (cp == END(buf) || *cp == '#')
	    vstring_truncate(buf, start);
	else if (start == 0 && lineno != 0 && first_line != 0)
	    *first_line = *lineno;
	/* Terminate at EOF or at the beginning of the next logical line. */
	if (ch == VSTREAM_EOF)
	    break;
	if (LEN(buf) > 0) {
	    if ((next = VSTREAM_GETC(fp)) != VSTREAM_EOF)
		vstream_ungetc(fp, next);
	    if (next != '#' && !ISSPACE(next))
		break;
	}
    }
    VSTRING_TERMINATE(buf);

    /*
     * Invalid input: continuing text without preceding text. Allowing this
     * would complicate "postconf -e", which implements its own multi-line
     * parsing routine. Do not abort, just warn, so that critical programs
     * like postmap do not leave behind a truncated table.
     */
    if (LEN(buf) > 0 && ISSPACE(*STR(buf))) {
	msg_warn("%s: logical line must not start with whitespace: \"%.30s%s\"",
		 VSTREAM_PATH(fp), STR(buf),
		 LEN(buf) > 30 ? "..." : "");
	return (readllines(buf, fp, lineno, first_line));
    }

    /*
     * Done.
     */
    return (LEN(buf) > 0 ? buf : 0);
}
