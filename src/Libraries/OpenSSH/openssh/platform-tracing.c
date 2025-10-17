/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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

#include <sys/types.h>
#ifdef HAVE_SYS_PROCCTL_H
#include <sys/procctl.h>
#endif
#if defined(HAVE_SYS_PRCTL_H)
#include <sys/prctl.h>	/* For prctl() and PR_SET_DUMPABLE */
#endif
#ifdef HAVE_SYS_PTRACE_H
#include <sys/ptrace.h>
#endif
#ifdef HAVE_PRIV_H
#include <priv.h> /* For setpflags() and __PROC_PROTECT  */
#endif
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "log.h"

void
platform_disable_tracing(int strict)
{
#if defined(HAVE_PROCCTL) && defined(PROC_TRACE_CTL)
	/* On FreeBSD, we should make this process untraceable */
	int disable_trace = PROC_TRACE_CTL_DISABLE;

	/*
	 * On FreeBSD, we should make this process untraceable.
	 * pid=0 means "this process" but some older kernels do not
	 * understand that so retry with our own pid before failing.
	 */
	if (procctl(P_PID, 0, PROC_TRACE_CTL, &disable_trace) == 0)
		return;
	if (procctl(P_PID, getpid(), PROC_TRACE_CTL, &disable_trace) == 0)
		return;
	if (strict)
		fatal("unable to make the process untraceable: %s",
		    strerror(errno));
#endif
#if defined(HAVE_PRCTL) && defined(PR_SET_DUMPABLE)
	/* Disable ptrace on Linux without sgid bit */
	if (prctl(PR_SET_DUMPABLE, 0) != 0 && strict)
		fatal("unable to make the process undumpable: %s",
		    strerror(errno));
#endif
#if defined(HAVE_SETPFLAGS) && defined(__PROC_PROTECT)
	/* On Solaris, we should make this process untraceable */
	if (setpflags(__PROC_PROTECT, 1) != 0 && strict)
		fatal("unable to make the process untraceable: %s",
		    strerror(errno));
#endif
#ifdef PT_DENY_ATTACH
	/* Mac OS X */
	if (ptrace(PT_DENY_ATTACH, 0, 0, 0) == -1 && strict)
		fatal("unable to set PT_DENY_ATTACH: %s", strerror(errno));
#endif
}
