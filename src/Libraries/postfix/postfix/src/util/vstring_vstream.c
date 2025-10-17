/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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
#include <stdio.h>
#include <string.h>

/* Application-specific. */

#include "msg.h"
#include "vstring.h"
#include "vstream.h"
#include "vstring_vstream.h"

 /*
  * Macro to return the last character added to a VSTRING, for consistency.
  */
#define VSTRING_GET_RESULT(vp) \
    (VSTRING_LEN(vp) > 0 ? vstring_end(vp)[-1] : VSTREAM_EOF)

/* vstring_get - read line from file, keep newline */

int     vstring_get(VSTRING *vp, VSTREAM *fp)
{
    int     c;

    VSTRING_RESET(vp);
    while ((c = VSTREAM_GETC(fp)) != VSTREAM_EOF) {
	VSTRING_ADDCH(vp, c);
	if (c == '\n')
	    break;
    }
    VSTRING_TERMINATE(vp);
    return (VSTRING_GET_RESULT(vp));
}

/* vstring_get_nonl - read line from file, strip newline */

int     vstring_get_nonl(VSTRING *vp, VSTREAM *fp)
{
    int     c;

    VSTRING_RESET(vp);
    while ((c = VSTREAM_GETC(fp)) != VSTREAM_EOF && c != '\n')
	VSTRING_ADDCH(vp, c);
    VSTRING_TERMINATE(vp);
    return (c == '\n' ? c : VSTRING_GET_RESULT(vp));
}

/* vstring_get_null - read null-terminated string from file */

int     vstring_get_null(VSTRING *vp, VSTREAM *fp)
{
    int     c;

    VSTRING_RESET(vp);
    while ((c = VSTREAM_GETC(fp)) != VSTREAM_EOF && c != 0)
	VSTRING_ADDCH(vp, c);
    VSTRING_TERMINATE(vp);
    return (c == 0 ? c : VSTRING_GET_RESULT(vp));
}

/* vstring_get_bound - read line from file, keep newline, up to bound */

int     vstring_get_bound(VSTRING *vp, VSTREAM *fp, ssize_t bound)
{
    int     c;

    if (bound <= 0)
	msg_panic("vstring_get_bound: invalid bound %ld", (long) bound);

    VSTRING_RESET(vp);
    while (bound-- > 0 && (c = VSTREAM_GETC(fp)) != VSTREAM_EOF) {
	VSTRING_ADDCH(vp, c);
	if (c == '\n')
	    break;
    }
    VSTRING_TERMINATE(vp);
    return (VSTRING_GET_RESULT(vp));
}

/* vstring_get_nonl_bound - read line from file, strip newline, up to bound */

int     vstring_get_nonl_bound(VSTRING *vp, VSTREAM *fp, ssize_t bound)
{
    int     c;

    if (bound <= 0)
	msg_panic("vstring_get_nonl_bound: invalid bound %ld", (long) bound);

    VSTRING_RESET(vp);
    while (bound-- > 0 && (c = VSTREAM_GETC(fp)) != VSTREAM_EOF && c != '\n')
	VSTRING_ADDCH(vp, c);
    VSTRING_TERMINATE(vp);
    return (c == '\n' ? c : VSTRING_GET_RESULT(vp));
}

/* vstring_get_null_bound - read null-terminated string from file */

int     vstring_get_null_bound(VSTRING *vp, VSTREAM *fp, ssize_t bound)
{
    int     c;

    if (bound <= 0)
	msg_panic("vstring_get_null_bound: invalid bound %ld", (long) bound);

    VSTRING_RESET(vp);
    while (bound-- > 0 && (c = VSTREAM_GETC(fp)) != VSTREAM_EOF && c != 0)
	VSTRING_ADDCH(vp, c);
    VSTRING_TERMINATE(vp);
    return (c == 0 ? c : VSTRING_GET_RESULT(vp));
}

#ifdef TEST

 /*
  * Proof-of-concept test program: copy the source to this module to stdout.
  */
#include <fcntl.h>

#define TEXT_VSTREAM    "vstring_vstream.c"

int     main(void)
{
    VSTRING *vp = vstring_alloc(1);
    VSTREAM *fp;

    if ((fp = vstream_fopen(TEXT_VSTREAM, O_RDONLY, 0)) == 0)
	msg_fatal("open %s: %m", TEXT_VSTREAM);
    while (vstring_fgets(vp, fp))
	vstream_fprintf(VSTREAM_OUT, "%s", vstring_str(vp));
    vstream_fclose(fp);
    vstream_fflush(VSTREAM_OUT);
    vstring_free(vp);
    return (0);
}

#endif
