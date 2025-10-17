/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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

#if _lib_readlink

NoN(readlink)

#else

#include "fakelink.h"

#include <error.h>

#ifndef ENOSYS
#define ENOSYS	EINVAL
#endif

int
readlink(const char* path, char* buf, int siz)
{
	int	fd;
	int	n;

	if (siz > sizeof(FAKELINK_MAGIC))
	{
		if ((fd = open(path, O_RDONLY|O_cloexec)) < 0)
			return -1;
		if (read(fd, buf, sizeof(FAKELINK_MAGIC)) == sizeof(FAKELINK_MAGIC) && !strcmp(buf, FAKELINK_MAGIC) && (n = read(fd, buf, siz)) > 0 && !buf[n - 1])
		{
			close(fd);
			return n;
		}
		close(fd);
	}
	errno = ENOSYS;
	return -1;
}

#endif
