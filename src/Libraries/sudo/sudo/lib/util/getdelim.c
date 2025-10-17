/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 26, 2024.
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

#ifndef HAVE_GETDELIM

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "sudo_compat.h"

ssize_t
sudo_getdelim(char **buf, size_t *bufsize, int delim, FILE *fp)
{
    char *cp, *ep;
    int ch;

    if (*buf == NULL || *bufsize == 0) {
	char *tmp = realloc(*buf, LINE_MAX);
	if (tmp == NULL)
	    return -1;
	*buf = tmp;
	*bufsize = LINE_MAX;
    }
    cp = *buf;
    ep = cp + *bufsize;

    do {
	if (cp + 1 >= ep) {
	    char *newbuf = reallocarray(*buf, *bufsize, 2);
	    if (newbuf == NULL)
		goto bad;
	    *bufsize *= 2;
	    cp = newbuf + (cp - *buf);
	    ep = newbuf + *bufsize;
	    *buf = newbuf;
	}
	if ((ch = getc(fp)) == EOF) {
	    if (feof(fp))
		break;
	    goto bad;
	}
	*cp++ = ch;
    } while (ch != delim);

    /* getdelim(3) should never return a length of 0. */
    if (cp != *buf) {
	*cp = '\0';
	return (ssize_t)(cp - *buf);
    }
bad:
    /* Error, push back what was read if possible. */
    while (cp > *buf) {
	if (ungetc(*cp--, fp) == EOF)
	    break;
    }
    return -1;
}
#endif /* HAVE_GETDELIM */
