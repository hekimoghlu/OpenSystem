/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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

/*
 * -last 3 arg open
 */

#include <ast.h>

#if !defined(open) || !defined(_ast_O_LOCAL)

NoN(open)

#else

#undef	open

extern int	open(const char*, int, ...);

#include <ls.h>
#include <error.h>

#ifdef O_NOCTTY
#include <ast_tty.h>
#endif

int
_ast_open(const char* path, int op, ...)
{
	int		fd;
	int		mode;
	int		save_errno;
	struct stat	st;
	va_list		ap;

	save_errno = errno;
	va_start(ap, op);
	mode = (op & O_CREAT) ? va_arg(ap, int) : S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH;
	va_end(ap);
	if (op & ~(_ast_O_LOCAL-1))
	{
		if (!(op & O_CREAT))
			op &= ~O_EXCL;
		for (;;)
		{
			if (op & O_TRUNC)
			{
				if ((op & O_EXCL) && !access(path, F_OK))
				{
					errno = EEXIST;
					return(-1);
				}
				if ((fd = creat(path, (op & O_EXCL) ? 0 : mode)) < 0)
					return(-1);
				if (op & O_EXCL)
				{
					if (fstat(fd, &st) || (st.st_mode & S_IPERM))
					{
						errno = EEXIST;
						close(fd);
						return(-1);
					}
#if _lib_fchmod
					if (mode && fchmod(fd, mode))
#else
					if (mode && chmod(path, mode))
#endif
						errno = save_errno;
				}
				if ((op & O_ACCMODE) == O_RDWR)
				{
					close(fd);
					op &= ~(O_CREAT|O_TRUNC);
					continue;
				}
			}
			else if ((fd = open(path, op & (_ast_O_LOCAL-1), mode)) < 0)
			{
				if (op & O_CREAT)
				{
					op |= O_TRUNC;
					continue;
				}
				return(-1);
			}
			else if ((op & O_APPEND) && lseek(fd, 0L, SEEK_END) == -1L)
				errno = save_errno;
#if O_NOCTTY
			if ((op & O_NOCTTY) && ioctl(fd, TIOCNOTTY, 0))
				errno = save_errno;
#endif
			break;
		}
	}
	else fd = open(path, op, mode);
	return(fd);
}

#endif
