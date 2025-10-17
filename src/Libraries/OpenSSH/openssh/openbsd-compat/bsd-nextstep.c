/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 5, 2025.
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
#include "includes.h"

#ifdef HAVE_NEXT
#include <errno.h>
#include <sys/wait.h>
#include "bsd-nextstep.h"

pid_t
posix_wait(int *status)
{
	union wait statusp;
	pid_t wait_pid;

	#undef wait			/* Use NeXT's wait() function */
	wait_pid = wait(&statusp);
	if (status)
		*status = (int) statusp.w_status;

	return (wait_pid);
}

int
tcgetattr(int fd, struct termios *t)
{
	return (ioctl(fd, TIOCGETA, t));
}

int
tcsetattr(int fd, int opt, const struct termios *t)
{
	struct termios localterm;

	if (opt & TCSASOFT) {
		localterm = *t;
		localterm.c_cflag |= CIGNORE;
		t = &localterm;
	}
	switch (opt & ~TCSASOFT) {
	case TCSANOW:
		return (ioctl(fd, TIOCSETA, t));
	case TCSADRAIN:
		return (ioctl(fd, TIOCSETAW, t));
	case TCSAFLUSH:
		return (ioctl(fd, TIOCSETAF, t));
	default:
		errno = EINVAL;
		return (-1);
	}
}

int tcsetpgrp(int fd, pid_t pgrp)
{
	return (ioctl(fd, TIOCSPGRP, &pgrp));
}

speed_t cfgetospeed(const struct termios *t)
{
	return (t->c_ospeed);
}

speed_t cfgetispeed(const struct termios *t)
{
	return (t->c_ispeed);
}

int
cfsetospeed(struct termios *t,int speed)
{
	t->c_ospeed = speed;
	return (0);
}

int
cfsetispeed(struct termios *t, int speed)
{
	t->c_ispeed = speed;
	return (0);
}
#endif /* HAVE_NEXT */
