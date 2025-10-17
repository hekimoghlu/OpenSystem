/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 28, 2024.
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
 * POSIX waitpid()
 *
 * pid < -1 WUNTRACED may not be fully supported
 * process group specifics ignored by non-{waitpid,wait4}
 */

#include <ast.h>
#include <wait.h>

#if _lib_waitpid

NoN(waitpid)

#else

#if _lib_wait4

struct rusage;

extern int	wait4(int, int*, int, struct rusage*);

pid_t
waitpid(pid_t pid, int* status, int flags)
{
	return(wait4(pid, status, flags, NiL));
}

#else

#undef	SIGCLD

#if _lib_wait3

extern int		wait3(int*, int, struct rusage*);

#else

#if _lib_wait2

#define wait3(s,f,u)	wait2(s,f)

extern int		wait2(int*, int);

#else

#include <sig.h>

#define wait3(s,f,u)	wait(s)

static int	caught;

static void
catch(sig)
int	sig;
{
	NoP(sig);
	caught = 1;
}

#endif

#endif

#include <error.h>

struct zombie
{
	struct zombie*	next;
	int		status;
	pid_t		pid;
};

pid_t
waitpid(pid_t pid, int* status, int flags)
{
	register struct zombie*	zp;
	register struct zombie*	pp;
	register int		p;
	int			s;
#if !_lib_wait2 && !_lib_wait3
#if !defined(SIGCLD)
	int			n;
	int			oerrno;
#endif
	Sig_handler_t		handler;
#endif

	static struct zombie*	zombies;

	pp = 0;
	zp = zombies;
	while (zp)
	{
		if (zp->pid >= 0 && (zp->pid == pid || pid <= 0))
		{
			if (pp) pp->next = zp->next;
			else zombies = zp->next;
			if (status) *status = zp->status;
			pid = zp->pid;
			free(zp);
			return(pid);
		}
	}
	if (pid > 0 && kill(pid, 0) < 0) return(-1);
	for (;;)
	{
#if !_lib_wait2 && !_lib_wait3
#if !defined(SIGCLD)
		oerrno = errno;
#endif
		if (flags & WNOHANG)
		{
			caught = 0;
#if defined(SIGCLD)
			handler = signal(SIGCLD, catch);
			if (!caught)
			{
				signal(SIGCLD, handler);
				return(0);
			}
#else
#if defined(SIGALRM)
			handler = signal(SIGALRM, catch);
			n = alarm(1);
#endif
#endif
		}
#endif
		p = wait3(&s, flags, NiL);
#if !_lib_wait3
#if !_lib_wait2
#if defined(SIGCLD)
		if (flags & WNOHANG) signal(SIGCLD, handler);
#else
#if defined(SIGALRM)
		if (flags & WNOHANG)
		{
			if (n == 0 && !caught || n == 1) alarm(n);
			else if (n > 1) alarm(n - caught);
			signal(SIGALRM, handler);
		}
		if (p == -1 && errno == EINTR)
		{
			errno = oerrno;
			p = 0;
			s = 0;
		}
#endif
#endif
#else
		if (p == -1 && errno == EINVAL && (flags & ~WNOHANG))
			p = wait3(&s, flags & WNOHANG, NiL);
#endif
#endif
		if (p <= 0)
		{
			if (p == 0 && status) *status = s;
			return(p);
		}
		if (pid <= 0 || p == pid)
		{
			if (status) *status = s;
			return(p);
		}
		if (!(zp = newof(0, struct zombie, 1, 0))) return(-1);
		zp->pid = p;
		zp->status = s;
		zp->next = zombies;
		zombies = zp;
	}
	/*NOTREACHED*/
}

#endif

#endif
