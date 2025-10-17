/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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
#include <config.h>

#ifndef HAVE_VSYSLOG

#include <stdio.h>
#include <syslog.h>
#include <stdarg.h>

#include "roken.h"

/*
 * the theory behind this is that we might be trying to call vsyslog
 * when there's no memory left, and we should try to be as useful as
 * possible.  And the format string should say something about what's
 * failing.
 */

static void
simple_vsyslog(int pri, const char *fmt, va_list ap)
{
    syslog (pri, "%s", fmt);
}

/*
 * do like syslog but with a `va_list'
 */

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
vsyslog(int pri, const char *fmt, va_list ap)
{
    char *fmt2;
    const char *p;
    char *p2;
    int ret;
    int saved_errno = errno;
    int fmt_len  = strlen (fmt);
    int fmt2_len = fmt_len;
    char *buf;

    fmt2 = malloc (fmt_len + 1);
    if (fmt2 == NULL) {
	simple_vsyslog (pri, fmt, ap);
	return;
    }

    for (p = fmt, p2 = fmt2; *p != '\0'; ++p) {
	if (p[0] == '%' && p[1] == 'm') {
	    const char *e = strerror (saved_errno);
	    int e_len = strlen (e);
	    char *tmp;
	    int pos;

	    pos = p2 - fmt2;
	    fmt2_len += e_len - 2;
	    tmp = realloc (fmt2, fmt2_len + 1);
	    if (tmp == NULL) {
		free (fmt2);
		simple_vsyslog (pri, fmt, ap);
		return;
	    }
	    fmt2 = tmp;
	    p2   = fmt2 + pos;
	    memmove (p2, e, e_len);
	    p2 += e_len;
	    ++p;
	} else
	    *p2++ = *p;
    }
    *p2 = '\0';

    ret = vasprintf (&buf, fmt2, ap);
    free (fmt2);
    if (ret < 0 || buf == NULL) {
	simple_vsyslog (pri, fmt, ap);
	return;
    }
    syslog (pri, "%s", buf);
    free (buf);
}
#endif
