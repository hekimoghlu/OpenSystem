/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 11, 2023.
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
#include <fcntl.h>
#include <stdbool.h>
#include <string.h>
#include <subsystem.h>
#include <sys/errno.h>
#include <sys/syslimits.h>
#include <_simple.h>

#define SUBSYSTEM_ROOT_PATH_KEY "subsystem_root_path"

void _subsystem_init(const char *apple[]);

static char * subsystem_root_path = NULL;
static size_t subsystem_root_path_len = 0;

/*
 * Takes the apple array, and initializes subsystem
 * support in Libc.
 */
void
_subsystem_init(const char **apple)
{
	char * subsystem_root_path_string = _simple_getenv(apple, SUBSYSTEM_ROOT_PATH_KEY);
	if (subsystem_root_path_string) {
		subsystem_root_path = subsystem_root_path_string;
		subsystem_root_path_len = strnlen(subsystem_root_path, PATH_MAX);
	}
}

/*
 * Takes a buffer, a subsystem path, and a file path, and constructs the
 * subsystem path for the given file path.  Assumes that the subsystem root
 * path will be "/" terminated.
 */
static bool
construct_subsystem_path(char * buf, size_t buf_size, const char * subsystem_root_path, const char * file_path)
{
	size_t return_a = strlcpy(buf, subsystem_root_path, buf_size);
	size_t return_b = strlcat(buf, file_path, buf_size);

	if ((return_a >= buf_size) || (return_b >= buf_size)) {
		return false;
	}

	return true;
}

int
open_with_subsystem(const char * path, int oflag)
{
	/* Don't support file creation. */
	if (oflag & O_CREAT){
		errno = EINVAL;
		return -1;
	}

	int result;

	result = open(path, oflag);

	if ((result < 0) && (errno == ENOENT) && (subsystem_root_path)) {
		/*
		 * If the file doesn't exist relative to root, search
		 * for it relative to the subsystem root.
		 */
		char subsystem_path[PATH_MAX];

		if (construct_subsystem_path(subsystem_path, sizeof(subsystem_path), subsystem_root_path, path)) {
			result = open(subsystem_path, oflag);
		} else {
			errno = ENAMETOOLONG;
		}
	}

	return result;
}

int
stat_with_subsystem(const char *restrict path, struct stat *restrict buf)
{
	int result;

	result = stat(path, buf);

	if ((result < 0) && (errno == ENOENT) && (subsystem_root_path)) {
		/*
		 * If the file doesn't exist relative to root, search
		 * for it relative to the subsystem root.
		 */
		char subsystem_path[PATH_MAX];

		if (construct_subsystem_path(subsystem_path, sizeof(subsystem_path), subsystem_root_path, path)) {
			result = stat(subsystem_path, buf);
		} else {
			errno = ENAMETOOLONG;
		}
	}

	return result;
}

