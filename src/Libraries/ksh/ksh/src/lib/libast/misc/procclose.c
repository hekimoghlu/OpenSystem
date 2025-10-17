/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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
 * close a proc opened by procopen()
 * otherwise exit() status of process is returned
 */

#include "proclib.h"

int
procclose(register Proc_t* p)
{
	int	pid;
	int	flags = 0;
	int	status = -1;

	if (p)
	{
		if (p->rfd >= 0)
			close(p->rfd);
		if (p->wfd >= 0 && p->wfd != p->rfd)
			close(p->wfd);
		if (p->flags & PROC_ORPHAN)
			status = 0;
		else
		{
			if (p->flags & PROC_ZOMBIE)
			{
				/*
				 * process may leave a zombie behind
				 * give it a chance to do that but
				 * don't hang waiting for it
				 */

				flags |= WNOHANG;
				sleep(1);
			}
			if (!(p->flags & PROC_FOREGROUND))
				sigcritical(SIG_REG_EXEC|SIG_REG_PROC);
			while ((pid = waitpid(p->pid, &status, flags)) == -1 && errno == EINTR);
			if (pid != p->pid && (flags & WNOHANG))
				status = 0;
			if (!(p->flags & PROC_FOREGROUND))
				sigcritical(0);
			else
			{
				if (p->sigint != SIG_IGN)
					signal(SIGINT, p->sigint);
				if (p->sigquit != SIG_IGN)
					signal(SIGQUIT, p->sigquit);
#if defined(SIGCHLD)
#if _lib_sigprocmask
				sigprocmask(SIG_SETMASK, &p->mask, NiL);
#else
#if _lib_sigsetmask
				sigsetmask(p->mask);
#else
				if (p->sigchld != SIG_DFL)
					signal(SIGCHLD, p->sigchld);
#endif
#endif
#endif
			}
			status = status == -1 ?
				 EXIT_QUIT :
				 WIFSIGNALED(status) ?
				 EXIT_TERM(WTERMSIG(status)) :
				 EXIT_CODE(WEXITSTATUS(status));
		}
		procfree(p);
	}
	else
		status = errno == ENOENT ? EXIT_NOTFOUND : EXIT_NOEXEC;
	return status;
}
