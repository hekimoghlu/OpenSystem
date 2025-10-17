/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 17, 2022.
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
#pragma prototyped

#include <ast.h>

#if _lib_symlink

NoN(symlink)

#else

#include "fakelink.h"

#include <error.h>

#ifndef ENOSYS
#define ENOSYS	EINVAL
#endif

int
symlink(const char* a, char* b)
{
	if (*a == '/' && (*(a + 1) == 'd' || *(a + 1) == 'p' || *(a + 1) == 'n') && (!strncmp(a, "/dev/tcp/", 9) || !strncmp(a, "/dev/udp/", 9) || !strncmp(a, "/proc/", 6) || !strncmp(a, "/n/", 3)))
	{
		int	n;
		int	fd;

		if ((fd = open(b, O_CREAT|O_TRUNC|O_WRONLY|O_cloexec, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH)) < 0)
			return -1;
		n = strlen(a) + 1;
		n = (write(fd, FAKELINK_MAGIC, sizeof(FAKELINK_MAGIC)) != sizeof(FAKELINK_MAGIC) || write(fd, a, n) != n) ? -1 : 0;
		close(fd);
		return n;
	}
	errno = ENOSYS;
	return -1;
}

#endif
