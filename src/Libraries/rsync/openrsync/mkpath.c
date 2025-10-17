/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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

#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdint.h>
#include <string.h>

#include "extern.h"

/*
 * We're using AT_RESOLVE_BENEATH in a couple of places just for some additional
 * safety on platforms that support it, so it's not a hard requirement.
 */
#ifndef AT_RESOLVE_BENEATH
#define	AT_RESOLVE_BENEATH	0
#endif

/* Code taken directly from mkdir(1).

 * mkpath -- create directories.
 *	path     - path
 */
int
mkpath(char *path, mode_t mode)
{
	struct stat sb;
	char *slash;
	int done;

	slash = path;

	for (;;) {
		slash += strspn(slash, "/");
		slash += strcspn(slash, "/");

		done = (*slash == '\0');
		*slash = '\0';

		if (mkdir(path, mode) != 0) {
			int mkdir_errno = errno;

			if (stat(path, &sb) == -1) {
				/* Not there; use mkdir()s errno */
				errno = mkdir_errno;
				*slash = done ? '\0' : '/';
				return (-1);
			}
			if (!S_ISDIR(sb.st_mode)) {
				/* Is there, but isn't a directory */
				errno = ENOTDIR;
				*slash = done ? '\0' : '/';
				return (-1);
			}
		}

		if (done)
			break;

		*slash = '/';
	}

	return (0);
}

int
mkpathat(int fd, char *path, mode_t mode)
{
	struct stat sb;
	char *slash;
	int done;

	slash = path;

	for (;;) {
		slash += strspn(slash, "/");
		slash += strcspn(slash, "/");

		done = (*slash == '\0');
		*slash = '\0';

		if (mkdirat(fd, path, mode) != 0) {
			int mkdir_errno = errno;

			if (fstatat(fd, path, &sb, AT_RESOLVE_BENEATH) == -1) {
				/* Not there; use mkdir()s errno */
				errno = mkdir_errno;
				*slash = '/';
				return (-1);
			}
			if (!S_ISDIR(sb.st_mode)) {
				/* Is there, but isn't a directory */
				errno = ENOTDIR;
				*slash = '/';
				return (-1);
			}
		}

		if (done)
			break;

		*slash = '/';
	}

	return (0);
}
