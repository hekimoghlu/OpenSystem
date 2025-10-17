/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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
#include "ruby/config.h"

#if defined(HAVE_FCNTL)
# include <fcntl.h>
#endif

#if !defined(HAVE_FCNTL) || !defined(F_DUPFD)
# include <errno.h>
#endif

#define BADEXIT -1

int
dup2(int fd1, int fd2)
{
#if defined(HAVE_FCNTL) && defined(F_DUPFD)
	if (fd1 != fd2) {
#ifdef F_GETFL
		if (fcntl(fd1, F_GETFL) < 0)
			return BADEXIT;
		if (fcntl(fd2, F_GETFL) >= 0)
			close(fd2);
#else
		close(fd2);
#endif
		if (fcntl(fd1, F_DUPFD, fd2) < 0)
			return BADEXIT;
	}
	return fd2;
#else
	extern int errno;
	int i, fd, fds[256];

	if (fd1 == fd2) return 0;
	close(fd2);
	for (i=0; i<256; i++) {
		fd = fds[i] = dup(fd1);
		if (fd == fd2) break;
	}
	while (i) {
	    	close(fds[i--]);
	}
	if (fd == fd2) return 0;
	errno = EMFILE;
	return BADEXIT;
#endif
}
