/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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
/* Roughly equivalent to "mktemp -d -t TEMPLATE", but portable. */

#include "includes.h"

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "log.h"

static void
usage(void)
{
	fprintf(stderr, "mkdtemp template\n");
	exit(1);
}

int
main(int argc, char **argv)
{
	const char *base;
	const char *tmpdir;
	char template[PATH_MAX];
	int r;
	char *dir;

	if (argc != 2)
		usage();
	base = argv[1];

	if ((tmpdir = getenv("TMPDIR")) == NULL)
		tmpdir = "/tmp";
	r = snprintf(template, sizeof(template), "%s/%s", tmpdir, base);
	if (r < 0 || (size_t)r >= sizeof(template))
		fatal("template string too long");
	dir = mkdtemp(template);
	if (dir == NULL) {
		perror("mkdtemp");
		exit(1);
	}
	puts(dir);
	return 0;
}
