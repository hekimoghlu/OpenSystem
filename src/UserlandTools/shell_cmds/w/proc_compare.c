/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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
#if 0
#ifndef lint
static char sccsid[] = "@(#)proc_compare.c	8.2 (Berkeley) 9/23/93";
#endif /* not lint */
#endif

#include <sys/cdefs.h>
#ifndef __APPLE__
__FBSDID("$FreeBSD$");
#endif

#include <sys/param.h>
#include <sys/proc.h>
#include <sys/time.h>
#include <sys/user.h>

#include "extern.h"

/*
 * Returns 1 if p2 is "better" than p1
 *
 * The algorithm for picking the "interesting" process is thus:
 *
 *	1) Only foreground processes are eligible - implied.
 *	2) Runnable processes are favored over anything else.  The runner
 *	   with the highest cpu utilization is picked (p_estcpu).  Ties are
 *	   broken by picking the highest pid.
 *	3) The sleeper with the shortest sleep time is next.  With ties,
 *	   we pick out just "short-term" sleepers (TDF_SINTR == 0).
 *	4) Further ties are broken by picking the highest pid.
 *
 * If you change this, be sure to consider making the change in the kernel
 * too (^T in kern/tty.c).
 *
 * TODO - consider whether pctcpu should be used.
 */

#if !HAVE_KVM
#define	ki_estcpu	p_estcpu
#define	ki_pid		p_pid
#define	ki_slptime	p_slptime
#define	ki_stat		p_stat
#define	ki_tdflags	p_tdflags
#endif

#define	ISRUN(p)	(((p)->ki_stat == SRUN) || ((p)->ki_stat == SIDL))
#define	TESTAB(a, b)    ((a)<<1 | (b))
#define	ONLYA   2
#define	ONLYB   1
#define	BOTH    3

#if HAVE_KVM
int
proc_compare(struct kinfo_proc *p1, struct kinfo_proc *p2)
{
#else
int
proc_compare(struct kinfo_proc *arg1, struct kinfo_proc *arg2)
{
	struct extern_proc* p1 = &arg1->kp_proc;
	struct extern_proc* p2 = &arg2->kp_proc;
#endif

	if (p1 == NULL)
		return (1);
	/*
	 * see if at least one of them is runnable
	 */
	switch (TESTAB(ISRUN(p1), ISRUN(p2))) {
	case ONLYA:
		return (0);
	case ONLYB:
		return (1);
	case BOTH:
		/*
		 * tie - favor one with highest recent cpu utilization
		 */
		if (p2->ki_estcpu > p1->ki_estcpu)
			return (1);
		if (p1->ki_estcpu > p2->ki_estcpu)
			return (0);
		return (p2->ki_pid > p1->ki_pid); /* tie - return highest pid */
	}
	/*
	 * weed out zombies
	 */
	switch (TESTAB(p1->ki_stat == SZOMB, p2->ki_stat == SZOMB)) {
	case ONLYA:
		return (1);
	case ONLYB:
		return (0);
	case BOTH:
		return (p2->ki_pid > p1->ki_pid); /* tie - return highest pid */
	}
	/*
	 * pick the one with the smallest sleep time
	 */
	if (p2->ki_slptime > p1->ki_slptime)
		return (0);
	if (p1->ki_slptime > p2->ki_slptime)
		return (1);
#ifndef __APPLE__
	/*
	 * favor one sleeping in a non-interruptible sleep
	 */
	if (p1->ki_tdflags & TDF_SINTR && (p2->ki_tdflags & TDF_SINTR) == 0)
		return (1);
	if (p2->ki_tdflags & TDF_SINTR && (p1->ki_tdflags & TDF_SINTR) == 0)
		return (0);
#endif
	return (p2->ki_pid > p1->ki_pid);	/* tie - return highest pid */
}
