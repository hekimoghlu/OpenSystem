/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
#include <sys/types.h>
#include <stdarg.h>
#include <sys/fcntl.h>
#include <sys/errno.h>
#include <sys/content_protection.h>

int __open_dprotected_np(const char* path, int flags, int class, int dpflags, int mode);
int __openat_dprotected_np(int fd, const char* path, int flags, int class, int dpflags, int mode, int authfd);

int
open_dprotected_np(const char *path, int flags, int class, int dpflags, ...)
{
	int mode = 0;

	if (dpflags & O_DP_AUTHENTICATE) {
		errno = EINVAL;
		return -1;
	}

	if (flags & O_CREAT) {
		va_list ap;
		va_start(ap, dpflags);
		mode = va_arg(ap, int);
		va_end(ap);
	}
	return __open_dprotected_np(path, flags, class, dpflags, mode);
}

int
openat_dprotected_np(int fd, const char *path, int flags, int class, int dpflags, ...)
{
	int mode = 0;

	if (dpflags & O_DP_AUTHENTICATE) {
		errno = EINVAL;
		return -1;
	}

	if (flags & O_CREAT) {
		va_list ap;
		va_start(ap, dpflags);
		mode = va_arg(ap, int);
		va_end(ap);
	}
	return __openat_dprotected_np(fd, path, flags, class, dpflags, mode, AUTH_OPEN_NOAUTHFD);
}

int
openat_authenticated_np(int fd, const char *path, int flags, int authfd)
{
	if (flags & O_CREAT) {
		errno = EINVAL;
		return -1;
	}

	if ((authfd != AUTH_OPEN_NOAUTHFD) && (authfd < 0)) {
		errno = EBADF;
		return -1;
	}

	return __openat_dprotected_np(fd, path, flags, PROTECTION_CLASS_DEFAULT, O_DP_AUTHENTICATE, 0, authfd);
}
