/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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

#if _lib_setsid

NoN(setsid)

#else

#include <ast_tty.h>
#include <error.h>

/*
 * become new process group leader and drop control tty
 */

pid_t
setsid(void)
{
	int	pg;
#ifdef TIOCNOTTY
	int	fd;
#endif

	/*
	 * become a new process group leader
	 */

	if ((pg = getpid()) == getpgrp())
	{
		errno = EPERM;
		return(-1);
	}
	setpgid(pg, pg);
#ifdef TIOCNOTTY

	/*
	 * drop the control tty
	 */

	if ((fd = open("/dev/tty", O_RDONLY|O_cloexec)) >= 0)
	{
		ioctl(fd, TIOCNOTTY, 0);
		close(fd);
	}
#else

	/*
	 * second child in s5 to avoid reacquiring the control tty
	 */

#if _lib_fork && HUH920711 /* some s5's botch this */
	switch (fork())
	{
	case -1:
		exit(1);
	case 0:
		break;
	default:
		exit(0);
	}
#endif

#endif
	return(pg);
}

#endif
