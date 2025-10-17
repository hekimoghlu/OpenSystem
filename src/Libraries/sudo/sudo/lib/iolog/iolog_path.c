/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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
/*
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif /* HAVE_STDBOOL_H */
#include <string.h>
#include <limits.h>
#include <time.h>
#include <unistd.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_fatal.h"
#include "sudo_gettext.h"
#include "sudo_iolog.h"
#include "sudo_util.h"

/*
 * Expand any escape sequences in inpath, returning the expanded path.
 */
bool
expand_iolog_path(const char *inpath, char *path, size_t pathlen,
    const struct iolog_path_escape *escapes, void *closure)
{
    char *dst, *pathend, tmpbuf[PATH_MAX];
    const char *endbrace, *src;
    bool strfit = false;
    size_t len;
    debug_decl(expand_iolog_path, SUDO_DEBUG_UTIL);

    /* Collapse multiple leading slashes. */
    while (inpath[0] == '/' && inpath[1] == '/')
	inpath++;

    pathend = path + pathlen;
    for (src = inpath, dst = path; *src != '\0'; src++) {
	if (src[0] == '%') {
	    if (src[1] == '{') {
		endbrace = strchr(src + 2, '}');
		if (endbrace != NULL) {
		    const struct iolog_path_escape *esc;
		    len = (size_t)(endbrace - src - 2);
		    for (esc = escapes; esc->name != NULL; esc++) {
			if (strncmp(src + 2, esc->name, len) == 0 &&
			    esc->name[len] == '\0')
			    break;
		    }
		    if (esc->name != NULL) {
			len = esc->copy_fn(dst, (size_t)(pathend - dst),
			    closure);
			if (len >= (size_t)(pathend - dst))
			    goto bad;
			dst += len;
			src = endbrace;
			continue;
		    }
		}
	    } else if (src[1] == '%') {
		/* Collapse %% -> % */
		src++;
	    } else {
		/* May need strftime() */
		strfit = true;
	    }
	}
	/* Need at least 2 chars, including the NUL terminator. */
	if (dst + 1 >= pathend)
	    goto bad;
	*dst++ = *src;
    }

    /* Trim trailing slashes and NUL terminate. */
    while (dst > path && dst[-1] == '/')
	dst--;
    *dst = '\0';

    /* Expand strftime escapes as needed. */
    if (strfit) {
	struct tm tm;
	time_t now;

	time(&now);
	if (localtime_r(&now, &tm) == NULL)
	    goto bad;

	/* We only call strftime() on the current part of the buffer. */
	tmpbuf[sizeof(tmpbuf) - 1] = '\0';
	len = strftime(tmpbuf, sizeof(tmpbuf), path, &tm);

	if (len == 0 || tmpbuf[sizeof(tmpbuf) - 1] != '\0')
	    goto bad;		/* strftime() failed, buf too small? */

	if (len >= (size_t)(pathend - path))
	    goto bad;		/* expanded buffer too big to fit. */
	memcpy(path, tmpbuf, len);
	dst = path + len;
	*dst = '\0';
    }

    debug_return_bool(true);
bad:
    debug_return_bool(false);
}
