/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 29, 2022.
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
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <paths.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include "util.h"

/*
 * opendev(3) is an inherently non-thread-safe API, since
 * it returns a buffer to global storage. However we can
 * at least make sure the storage allocation is thread safe
 * and does not leak memory in case of simultaneous
 * initialization
 */
static pthread_once_t opendev_namebuf_once = PTHREAD_ONCE_INIT;
static char *namebuf = NULL;

static void opendev_namebuf_init(void);

int
opendev(path, oflags, dflags, realpath)
	char *path;
	int oflags;
	int dflags;
	char **realpath;
{
	int fd;
	char *slash, *prefix;

	/* Initial state */
	if (realpath)
		*realpath = path;
	fd = -1;
	errno = ENOENT;

	if (pthread_once(&opendev_namebuf_once,
					 opendev_namebuf_init)
		|| !namebuf) {
		errno = ENOMEM;
		return -1;
	}

	if (dflags & OPENDEV_BLCK)
		prefix = "";			/* block device */
	else
		prefix = "r";			/* character device */

	if ((slash = strchr(path, '/')))
		fd = open(path, oflags);
	else if (dflags & OPENDEV_PART) {
		if (snprintf(namebuf, PATH_MAX, "%s%s%s",
		    _PATH_DEV, prefix, path) < PATH_MAX) {
			char *slice;
			while ((slice = strrchr(namebuf, 's')) &&
			    isdigit(*(slice-1))) *slice = '\0';
			fd = open(namebuf, oflags);
			if (realpath)
				*realpath = namebuf;
		} else
			errno = ENAMETOOLONG;
	}
	if (!slash && fd == -1 && errno == ENOENT) {
		if (snprintf(namebuf, PATH_MAX, "%s%s%s",
		    _PATH_DEV, prefix, path) < PATH_MAX) {
			fd = open(namebuf, oflags);
			if (realpath)
				*realpath = namebuf;
		} else
			errno = ENAMETOOLONG;
	}
	return (fd);
}

static void opendev_namebuf_init(void)
{
	namebuf = malloc(PATH_MAX);
}
