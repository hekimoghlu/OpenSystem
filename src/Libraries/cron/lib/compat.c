/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 3, 2024.
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
#if !defined(lint) && !defined(LINT)
static char rcsid[] = "$FreeBSD: src/usr.sbin/cron/lib/compat.c,v 1.7 2000/12/09 09:35:44 obrien Exp $";
#endif

/* vix 30dec93 [broke this out of misc.c - see RCS log for history]
 * vix 15jan87 [added TIOCNOTTY, thanks csg@pyramid]
 */


#include "cron.h"
#ifdef NEED_GETDTABLESIZE
# include <limits.h>
#endif
#if defined(NEED_SETSID) && defined(BSD)
# include <sys/ioctl.h>
#endif
#include <errno.h>
#include <paths.h>


/* the code does not depend on any of vfork's
 * side-effects; it just uses it as a quick
 * fork-and-exec.
 */
#ifdef NEED_VFORK
PID_T
vfork() {
	return (fork());
}
#endif


#ifdef NEED_STRDUP
char *
strdup(str)
	char	*str;
{
	char	*temp;

	if ((temp = malloc(strlen(str) + 1)) == NULL) {
		errno = ENOMEM;
		return NULL;
	}
	(void) strcpy(temp, str);
	return temp;
}
#endif


#ifdef NEED_STRERROR
char *
strerror(error)
	int error;
{
	extern char *sys_errlist[];
	extern int sys_nerr;
	static char buf[32];

	if ((error <= sys_nerr) && (error > 0)) {
		return sys_errlist[error];
	}

	sprintf(buf, "Unknown error: %d", error);
	return buf;
}
#endif


#ifdef NEED_STRCASECMP
int
strcasecmp(left, right)
	char	*left;
	char	*right;
{
	while (*left && (MkLower(*left) == MkLower(*right))) {
		left++;
		right++;
	}
	return MkLower(*left) - MkLower(*right);
}
#endif


#ifdef NEED_SETSID
int
setsid()
{
	int	newpgrp;
# if defined(BSD)
	int	fd;
#  if defined(POSIX)
	newpgrp = setpgid((pid_t)0, getpid());
#  else
	newpgrp = setpgrp(0, getpid());
#  endif
	if ((fd = open(_PATH_TTY, 2)) >= 0)
	{
		(void) ioctl(fd, TIOCNOTTY, (char*)0);
		(void) close(fd);
	}
# else /*BSD*/
	newpgrp = setpgrp();

	(void) close(STDIN);	(void) open(_PATH_DEVNULL, 0);
	(void) close(STDOUT);	(void) open(_PATH_DEVNULL, 1);
	(void) close(STDERR);	(void) open(_PATH_DEVNULL, 2);
# endif /*BSD*/
	return newpgrp;
}
#endif /*NEED_SETSID*/


#ifdef NEED_GETDTABLESIZE
int
getdtablesize() {
#ifdef _SC_OPEN_MAX
	return sysconf(_SC_OPEN_MAX);
#else
	return _POSIX_OPEN_MAX;
#endif
}
#endif


#ifdef NEED_FLOCK
/* The following flock() emulation snarfed intact *) from the HP-UX
 * "BSD to HP-UX porting tricks" maintained by
 * system@alchemy.chem.utoronto.ca (System Admin (Mike Peterson))
 * from the version "last updated: 11-Jan-1993"
 * Snarfage done by Jarkko Hietaniemi <Jarkko.Hietaniemi@hut.fi>
 * *) well, almost, had to K&R the function entry, HPUX "cc"
 * does not grok ANSI function prototypes */
 
/*
 * flock (fd, operation)
 *
 * This routine performs some file locking like the BSD 'flock'
 * on the object described by the int file descriptor 'fd',
 * which must already be open.
 *
 * The operations that are available are:
 *
 * LOCK_SH  -  get a shared lock.
 * LOCK_EX  -  get an exclusive lock.
 * LOCK_NB  -  don't block (must be ORed with LOCK_SH or LOCK_EX).
 * LOCK_UN  -  release a lock.
 *
 * Return value: 0 if lock successful, -1 if failed.
 *
 * Note that whether the locks are enforced or advisory is
 * controlled by the presence or absence of the SETGID bit on
 * the executable.
 *
 * Note that there is no difference between shared and exclusive
 * locks, since the 'lockf' system call in SYSV doesn't make any
 * distinction.
 *
 * The file "<sys/file.h>" should be modified to contain the definitions
 * of the available operations, which must be added manually (see below
 * for the values).
 */

/* this code has been reformatted by vixie */

int
flock(fd, operation)
	int fd;
	int operation;
{
	int i;

	switch (operation) {
	case LOCK_SH:		/* get a shared lock */
	case LOCK_EX:		/* get an exclusive lock */
		i = lockf (fd, F_LOCK, 0);
		break;

	case LOCK_SH|LOCK_NB:	/* get a non-blocking shared lock */
	case LOCK_EX|LOCK_NB:	/* get a non-blocking exclusive lock */
		i = lockf (fd, F_TLOCK, 0);
		if (i == -1)
			if ((errno == EAGAIN) || (errno == EACCES))
				errno = EWOULDBLOCK;
		break;

	case LOCK_UN:		/* unlock */
		i = lockf (fd, F_ULOCK, 0);
		break;
 
	default:		/* can't decipher operation */
		i = -1;
		errno = EINVAL;
		break;
	}
 
	return (i);
}
#endif /*NEED_FLOCK*/


#ifdef NEED_SETENV
int
setenv(name, value, overwrite)
	char *name, *value;
	int overwrite;
{
	char *tmp;

	if (overwrite && getenv(name))
		return -1;

	if (!(tmp = malloc(strlen(name) + strlen(value) + 2))) {
		errno = ENOMEM;
		return -1;
	}

	sprintf(tmp, "%s=%s", name, value);
	return putenv(tmp);	/* intentionally orphan 'tmp' storage */
}
#endif
