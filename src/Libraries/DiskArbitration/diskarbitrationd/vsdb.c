/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 1, 2023.
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
#include "vsdb.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static FILE *_vs_fp;
static struct vsdb _vs_vsdb;

static int
vsdbscan()
{
	char *cp, *p;
#define	MAXLINELENGTH	1024
	static char line[MAXLINELENGTH];

	for (;;) {

		if (!(p = fgets(line, sizeof(line), _vs_fp)))
			return(0);
		if (!(cp = strsep(&p, ":")) || *cp == '\0')
			continue;
		_vs_vsdb.vs_spec = cp;
		if (!(cp = strsep(&p, "\n")) || *cp == '\0')
			continue;
		_vs_vsdb.vs_ops = strtol(cp, &p, 16);
		if (*p == '\0')
			return(1);
	}
	/* NOTREACHED */
}

struct vsdb *
getvsent()
{
	if ((!_vs_fp && !setvsent()) || !vsdbscan())
		return((struct vsdb *)NULL);
	return(&_vs_vsdb);
}

struct vsdb *
getvsspec(name)
	const char *name;
{
	if (setvsent())
		while (vsdbscan())
			if (!strcmp(_vs_vsdb.vs_spec, name))
				return(&_vs_vsdb);
	return((struct vsdb *)NULL);
}

int 
setvsent()
{
	if (_vs_fp) {
		rewind(_vs_fp);
		return(1);
	}
	if ((_vs_fp = fopen(_PATH_VSDB, "r")) != NULL) {
		return(1);
	}
	return(0);
}

void
endvsent()
{
	if (_vs_fp) {
		(void)fclose(_vs_fp);
		_vs_fp = NULL;
	}
}
