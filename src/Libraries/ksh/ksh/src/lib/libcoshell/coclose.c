/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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
 * close a coshell
 */

#include "colib.h"

/*
 * called when coshell is hung
 */

static void
hung(int sig)
{
	NoP(sig);
	kill(state.current->pid, SIGKILL);
}

/*
 * shut down one coshell
 */

static int
shut(register Coshell_t* co)
{
	register Coshell_t*	cs;
	int			n;
	int			status;
	Coshell_t*		ps;
	Coservice_t*		sv;
	Sig_handler_t		handler;

	sfclose(co->msgfp);
	close(co->cmdfd);
	if (co->pid)
	{
		if (co->running > 0)
			killpg(co->pid, SIGTERM);
		state.current = co;
		handler = signal(SIGALRM, hung);
		n = alarm(3);
		if (waitpid(co->pid, &status, 0) != co->pid)
			status = -1;
		alarm(n);
		signal(SIGALRM, handler);
		killpg(co->pid, SIGTERM);
	}
	else
		status = 0;
	if (co->flags & CO_DEBUG)
		errormsg(state.lib, 2, "coshell %d jobs %d user %s sys %s", co->index, co->total, fmtelapsed(co->user, CO_QUANT), fmtelapsed(co->sys, CO_QUANT));
	for (sv = co->service; sv; sv = sv->next)
	{
		if (sv->fd > 0)
			close(sv->fd);
		if (sv->pid)
			waitpid(sv->pid, &status, 0);
	}
	cs = state.coshells;
	ps = 0;
	while (cs)
	{
		if (cs == co)
		{
			cs = cs->next;
			if (ps)
				ps->next = cs;
			else
				state.coshells = cs;
			vmclose(co->vm);
			break;
		}
		ps = cs;
		cs = cs->next;
	}
	return status;
}

/*
 * close coshell co
 */

int
coclose(register Coshell_t* co)
{
	if (co)
		return shut(co);
	while (state.coshells)
		shut(state.coshells);
	return 0;
}
