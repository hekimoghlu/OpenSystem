/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)termios.c	8.2 (Berkeley) 2/21/94";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/termios.c,v 1.16 2009/05/07 13:49:48 ed Exp $");

#if __DARWIN_UNIX03
#ifdef VARIANT_CANCELABLE
#include <pthread.h>
#endif /* VARIANT_CANCELABLE */
#endif /* __DARWIN_UNIX03 */

#include "namespace.h"
#include <sys/types.h>
#include <sys/fcntl.h>
#include <sys/ioctl.h>
#include <sys/time.h>

#include <errno.h>
#include <termios.h>
#include <unistd.h>
#include "un-namespace.h"

#ifndef BUILDING_VARIANT
int
tcgetattr(int fd, struct termios *t)
{

	return (_ioctl(fd, TIOCGETA, t));
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
		return (_ioctl(fd, TIOCSETA, t));
	case TCSADRAIN:
		return (_ioctl(fd, TIOCSETAW, t));
	case TCSAFLUSH:
		return (_ioctl(fd, TIOCSETAF, t));
	default:
		errno = EINVAL;
		return (-1);
	}
}

int
tcsetpgrp(int fd, pid_t pgrp)
{
	int s;

	if (isatty(fd) == 0)
		return (-1);

	s = pgrp;
	return (_ioctl(fd, TIOCSPGRP, &s));
}

pid_t
tcgetpgrp(int fd)
{
	int s;

	if (isatty(fd) == 0)
		return ((pid_t)-1);

	if (_ioctl(fd, TIOCGPGRP, &s) < 0)
		return ((pid_t)-1);

	return ((pid_t)s);
}

#if 0 // Needs API review first
pid_t
tcgetsid(int fd)
{
	int s;

	if (_ioctl(fd, TIOCGSID, &s) < 0)
		return ((pid_t)-1);

	return ((pid_t)s);
}

int
tcsetsid(int fd, pid_t pid)
{

	if (pid != getsid(0)) {
		errno = EINVAL;
		return (-1);
	}

	return (_ioctl(fd, TIOCSCTTY, NULL));
}
#endif

speed_t
cfgetospeed(const struct termios *t)
{

	return (t->c_ospeed);
}

speed_t
cfgetispeed(const struct termios *t)
{

	return (t->c_ispeed);
}

int
cfsetospeed(struct termios *t, speed_t speed)
{

	t->c_ospeed = speed;
	return (0);
}

int
cfsetispeed(struct termios *t, speed_t speed)
{

	t->c_ispeed = speed;
	return (0);
}

int
cfsetspeed(struct termios *t, speed_t speed)
{

	t->c_ispeed = t->c_ospeed = speed;
	return (0);
}

/*
 * Make a pre-existing termios structure into "raw" mode: character-at-a-time
 * mode with no characters interpreted, 8-bit data path.
 */
void
cfmakeraw(struct termios *t)
{

	t->c_iflag &= ~(IMAXBEL|IXOFF|INPCK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL|IXON|IGNPAR);
	t->c_iflag |= IGNBRK;
	t->c_oflag &= ~OPOST;
	t->c_lflag &= ~(ECHO|ECHOE|ECHOK|ECHONL|ICANON|ISIG|IEXTEN|NOFLSH|TOSTOP|PENDIN);
	t->c_cflag &= ~(CSIZE|PARENB);
	t->c_cflag |= CS8|CREAD;
	t->c_cc[VMIN] = 1;
	t->c_cc[VTIME] = 0;
}

int
tcsendbreak(int fd, int len)
{
	struct timeval sleepytime;

	sleepytime.tv_sec = 0;
	sleepytime.tv_usec = 400000;
	if (_ioctl(fd, TIOCSBRK, 0) == -1)
		return (-1);
	(void)_select(0, 0, 0, 0, &sleepytime);
	if (_ioctl(fd, TIOCCBRK, 0) == -1)
		return (-1);
	return (0);
}
#endif /* BUILDING_VARIANT */

int
__tcdrain(int fd)
{
#if __DARWIN_UNIX03
#ifdef VARIANT_CANCELABLE
	pthread_testcancel();
#endif /* VARIANT_CANCELABLE */
#endif /* __DARWIN_UNIX03 */
	return (_ioctl(fd, TIOCDRAIN, 0));
}

__weak_reference(__tcdrain, tcdrain);
__weak_reference(__tcdrain, _tcdrain);

#ifndef BUILDING_VARIANT
int
tcflush(int fd, int which)
{
	int com;

	switch (which) {
	case TCIFLUSH:
		com = FREAD;
		break;
	case TCOFLUSH:
		com = FWRITE;
		break;
	case TCIOFLUSH:
		com = FREAD | FWRITE;
		break;
	default:
		errno = EINVAL;
		return (-1);
	}
	return (_ioctl(fd, TIOCFLUSH, &com));
}

int
tcflow(int fd, int action)
{
	switch (action) {
	case TCOOFF:
		return (_ioctl(fd, TIOCSTOP, 0));
	case TCOON:
		return (_ioctl(fd, TIOCSTART, 0));
	case TCION:
		return (_ioctl(fd, TIOCIXON, 0));
	case TCIOFF:
		return (_ioctl(fd, TIOCIXOFF, 0));
	default:
		errno = EINVAL;
		return (-1);
	}
	/* NOTREACHED */
}
#endif /* BUILDING_VARIANT */
