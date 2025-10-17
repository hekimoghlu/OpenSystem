/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 7, 2021.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/usr.sbin/mtree/excludes.c,v 1.8 2003/10/21 08:27:05 phk Exp $");

#include <sys/types.h>
#include <sys/queue.h>

#include <err.h>
#include <errno.h>
#include <fnmatch.h>
#include <fts.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "metrics.h"
#include "extern.h"

/*
 * We're assuming that there won't be a whole lot of excludes,
 * so it's OK to use a stupid algorithm.
 */
struct exclude {
	LIST_ENTRY(exclude) link;
	const char *glob;
	int pathname;
};
static LIST_HEAD(, exclude) excludes;

void
init_excludes(void)
{
	LIST_INIT(&excludes);
}

void
read_excludes_file(const char *name)
{
	FILE *fp;
	char *line, *str;
	struct exclude *e;
	size_t len;

	fp = fopen(name, "r");
	if (fp == 0)
		err(1, "%s", name);

	while ((line = fgetln(fp, &len)) != 0) {
		if (line[len - 1] == '\n')
			len--;
		if (len == 0)
			continue;

		str = malloc(len + 1);
		e = malloc(sizeof *e);
		if (str == 0 || e == 0) {
			RECORD_FAILURE(59, ENOMEM);
			errx(1, "memory allocation error");
		}
		e->glob = str;
		memcpy(str, line, len);
		str[len] = '\0';
		if (strchr(str, '/'))
			e->pathname = 1;
		else
			e->pathname = 0;
		LIST_INSERT_HEAD(&excludes, e, link);
	}
	fclose(fp);
}

int
check_excludes(const char *fname, const char *path)
{
	struct exclude *e;

	/* fnmatch(3) has a funny return value convention... */
#define MATCH(g, n) (fnmatch((g), (n), FNM_PATHNAME) == 0)

	LIST_FOREACH(e, &excludes, link) {
		if ((e->pathname && MATCH(e->glob, path))
		    || MATCH(e->glob, fname))
			return 1;
	}
	return 0;
}
