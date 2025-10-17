/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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
 * -last fcntl
 */

#include <ast.h>

#ifndef fcntl

NoN(fcntl)

#else

#include <ls.h>
#include <ast_tty.h>
#include <error.h>

#if F_SETFD >= _ast_F_LOCAL
#if _sys_filio
#include <sys/filio.h>
#endif
#endif

#if _lib_fcntl
#undef	fcntl
extern int	fcntl(int, int, ...);
#endif

int
_ast_fcntl(int fd, int op, ...)
{
	int		n;
	int		save_errno;
	struct stat	st;
	va_list		ap;

	save_errno = errno;
	va_start(ap, op);
	if (op >= _ast_F_LOCAL) switch (op)
	{
#if F_DUPFD >= _ast_F_LOCAL
	case F_DUPFD:
		n = va_arg(ap, int);
		op = dup2(fd, n);
		break;
#endif
#if F_GETFL >= _ast_F_LOCAL
	case F_GETFL:
		op = fstat(fd, &st);
		break;
#endif
#if F_SETFD >= _ast_F_LOCAL && defined(FIOCLEX)
	case F_SETFD:
		n = va_arg(ap, int);
		op = ioctl(fd, n == FD_CLOEXEC ? FIOCLEX : FIONCLEX, 0);
		break;
#endif
	default:
		errno = EINVAL;
		op = -1;
		break;
	}
	else
#if _lib_fcntl
	op = fcntl(fd, op, va_arg(ap, int));
#else
	{
		errno = EINVAL;
		op = -1;
	}
#endif
	va_end(ap);
	return(op);
}

#endif
