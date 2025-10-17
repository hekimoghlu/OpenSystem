/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 29, 2022.
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
#include "config.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>
#include <sys/param.h>

#include "extern.h"

/*
 * Allocate space for a readlink(2) invocation.
 * Returns NULL on failure or a buffer otherwise.
 * The buffer must be passed to free() by the caller.
 */
char *
symlink_read(const char *path, size_t sz)
{
	char	*buf = NULL;
	ssize_t	 nsz = 0;
	void	*pp;

	while (true) {
		if ((pp = realloc(buf, sz + 1)) == NULL) {
			ERR("realloc");
			free(buf);
			return NULL;
		}
		buf = pp;

		if ((nsz = readlink(path, buf, sz + 1)) == -1) {
			ERR("%s: readlink", path);
			free(buf);
			return NULL;
		} else if (nsz == 0) {
			ERRX("%s: empty link", path);
			free(buf);
			return NULL;
		} else if ((size_t)nsz < sz + 1)
			break;

		sz = roundup(sz + 1, PATH_MAX);
	}

	assert(buf != NULL);
	assert(nsz > 0);
	buf[nsz] = '\0';
	return buf;
}

/*
 * Allocate space for a readlinkat(2) invocation.
 * Returns NULL on failure or a buffer otherwise.
 * The buffer must be passed to free() by the caller.
 */
char *
symlinkat_read(int fd, const char *path, size_t sz)
{
	char	*buf = NULL;
	ssize_t	 nsz = 0;
	void	*pp;

	while (true) {
		if ((pp = realloc(buf, sz + 1)) == NULL) {
			ERR("realloc");
			free(buf);
			return NULL;
		}
		buf = pp;

		if ((nsz = readlinkat(fd, path, buf, sz + 1)) == -1) {
			ERR("%s: readlinkat", path);
			free(buf);
			return NULL;
		} else if (nsz == 0) {
			ERRX("%s: empty link", path);
			free(buf);
			return NULL;
		} else if ((size_t)nsz < sz + 1)
			break;

		sz = roundup(sz + 1, PATH_MAX);
	}

	assert(buf != NULL);
	assert(nsz > 0);
	buf[nsz] = '\0';
	return buf;
}
