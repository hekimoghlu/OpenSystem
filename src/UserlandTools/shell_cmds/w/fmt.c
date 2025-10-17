/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 3, 2025.
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
#if 0
#ifndef lint
static char sccsid[] = "@(#)fmt.c	8.4 (Berkeley) 4/15/94";
#endif
#endif

#include <sys/cdefs.h>
#ifndef __APPLE__
__FBSDID("$FreeBSD$");
#endif

#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <err.h>
#include <limits.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vis.h>

#ifndef __APPLE__
#include "ps.h"
#endif

static char *cmdpart(char *);
static char *shquote(char **);

static char *
shquote(char **argv)
{
	long arg_max;
	static size_t buf_size;
	size_t len;
	char **p, *dst, *src;
	static char *buf = NULL;

	if (buf == NULL) {
		if ((arg_max = sysconf(_SC_ARG_MAX)) == -1)
			errx(1, "sysconf _SC_ARG_MAX failed");
		if (arg_max >= LONG_MAX / 4 || arg_max >= (long)(SIZE_MAX / 4))
			errx(1, "sysconf _SC_ARG_MAX preposterously large");
		buf_size = 4 * arg_max + 1;
		if ((buf = malloc(buf_size)) == NULL)
			errx(1, "malloc failed");
	}

	if (*argv == NULL) {
		buf[0] = '\0';
		return (buf);
	}
	dst = buf;
	for (p = argv; (src = *p++) != NULL; ) {
		if (*src == '\0')
			continue;
		len = (buf_size - 1 - (dst - buf)) / 4;
		strvisx(dst, src, strlen(src) < len ? strlen(src) : len,
		    VIS_NL | VIS_CSTYLE);
		while (*dst != '\0')
			dst++;
		if ((buf_size - 1 - (dst - buf)) / 4 > 0)
			*dst++ = ' ';
	}
	/* Chop off trailing space */
	if (dst != buf && dst[-1] == ' ')
		dst--;
	*dst = '\0';
	return (buf);
}

static char *
cmdpart(char *arg0)
{
	char *cp;

	return ((cp = strrchr(arg0, '/')) != NULL ? cp + 1 : arg0);
}

const char *
fmt_argv(char **argv, char *cmd, char *thread, size_t maxlen)
{
	size_t len;
	char *ap, *cp;

	if (argv == NULL || argv[0] == NULL) {
		if (cmd == NULL)
			return ("");
		ap = NULL;
		len = maxlen + 3;
	} else {
		ap = shquote(argv);
		len = strlen(ap) + maxlen + 4;
	}
	cp = malloc(len);
	if (cp == NULL)
		errx(1, "malloc failed");
	if (ap == NULL) {
		if (thread != NULL) {
			asprintf(&ap, "%s/%s", cmd, thread);
			sprintf(cp, "[%.*s]", (int)maxlen, ap);
			free(ap);
		} else
			sprintf(cp, "[%.*s]", (int)maxlen, cmd);
	} else if (strncmp(cmdpart(argv[0]), cmd, maxlen) != 0)
		sprintf(cp, "%s (%.*s)", ap, (int)maxlen, cmd);
	else
		strcpy(cp, ap);
	return (cp);
}
