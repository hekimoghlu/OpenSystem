/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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
 * Glenn Fowler
 * AT&T Research
 *
 * signal critical region support
 */

#include <ast.h>
#include <sig.h>

static struct
{
	int	sig;
	int	op;
}
signals[] =		/* held inside critical region	*/
{
	SIGINT,		SIG_REG_EXEC,
#ifdef SIGPIPE
	SIGPIPE,	SIG_REG_EXEC,
#endif
#ifdef SIGQUIT
	SIGQUIT,	SIG_REG_EXEC,
#endif
#ifdef SIGHUP
	SIGHUP,		SIG_REG_EXEC,
#endif
#if defined(SIGCHLD) && ( !defined(SIGCLD) || SIGCHLD != SIGCLD || _lib_sigprocmask || _lib_sigsetmask )
	SIGCHLD,	SIG_REG_PROC,
#endif
#ifdef SIGTSTP
	SIGTSTP,	SIG_REG_TERM,
#endif
#ifdef SIGTTIN
	SIGTTIN,	SIG_REG_TERM,
#endif
#ifdef SIGTTOU
	SIGTTOU,	SIG_REG_TERM,
#endif
};

#ifndef SIG_SETMASK
#undef	_lib_sigprocmask
#endif

#if !_lib_sigprocmask && !_lib_sigsetmask

static long	hold;			/* held signal mask		*/

/*
 * hold last signal for later delivery
 */

static void
interrupt(int sig)
{
	signal(sig, interrupt);
	hold |= sigmask(sig);
}

#endif

/*
 * critical signal region handler
 *
 * op>0		new region according to SIG_REG_*, return region level
 * op==0	pop region, return region level
 * op<0		return non-zero if any signals held in current region
 *
 * signals[] held until region popped
 */

int
sigcritical(int op)
{
	register int		i;
	static int		region;
	static int		level;
#if _lib_sigprocmask
	static sigset_t		mask;
	sigset_t		nmask;
#else
#if _lib_sigsetmask
	static long		mask;
#else
	static Sig_handler_t	handler[elementsof(signals)];
#endif
#endif

	if (op > 0)
	{
		if (!level++)
		{
			region = op;
			if (op & SIG_REG_SET)
				level--;
#if _lib_sigprocmask
			sigemptyset(&nmask);
			for (i = 0; i < elementsof(signals); i++)
				if (op & signals[i].op)
					sigaddset(&nmask, signals[i].sig);
			sigprocmask(SIG_BLOCK, &nmask, &mask);
#else
#if _lib_sigsetmask
			mask = 0;
			for (i = 0; i < elementsof(signals); i++)
				if (op & signals[i].op)
					mask |= sigmask(signals[i].sig);
			mask = sigblock(mask);
#else
			hold = 0;
			for (i = 0; i < elementsof(signals); i++)
				if ((op & signals[i].op) && (handler[i] = signal(signals[i].sig, interrupt)) == SIG_IGN)
				{
					signal(signals[i].sig, handler[i]);
					hold &= ~sigmask(signals[i].sig);
				}
#endif
#endif
		}
		return level;
	}
	else if (op < 0)
	{
#if _lib_sigprocmask
		sigpending(&nmask);
		for (i = 0; i < elementsof(signals); i++)
			if (region & signals[i].op)
			{
				if (sigismember(&nmask, signals[i].sig))
					return 1;
			}
		return 0;
#else
#if _lib_sigsetmask
		/* no way to get pending signals without installing handler */
		return 0;
#else
		return hold != 0;
#endif
#endif
	}
	else
	{
		/*
		 * a vfork() may have intervened so we
		 * allow apparent nesting mismatches
		 */

		if (--level <= 0)
		{
			level = 0;
#if _lib_sigprocmask
			sigprocmask(SIG_SETMASK, &mask, NiL);
#else
#if _lib_sigsetmask
			sigsetmask(mask);
#else
			for (i = 0; i < elementsof(signals); i++)
				if (region & signals[i].op)
					signal(signals[i].sig, handler[i]);
			if (hold)
			{
				for (i = 0; i < elementsof(signals); i++)
					if (region & signals[i].op)
					{
						if (hold & sigmask(signals[i].sig))
							kill(getpid(), signals[i].sig);
					}
				pause();
			}
#endif
#endif
		}
		return level;
	}
}
