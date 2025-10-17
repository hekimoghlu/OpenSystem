/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 21, 2025.
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
/*
 * Copyright (c) 1995 NeXT Computer, Inc. All Rights Reserved
 *
 *	@(#)sigaction.c	1.0
 */

#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <sys/signal.h>
#include <errno.h>

// keep in sync with BSD_KERNEL_PRIVATE value in xnu/bsd/sys/signal.h
#define SA_VALIDATE_SIGRETURN_FROM_SIGTRAMP 0x0400

/*
 *	Intercept the sigaction syscall and use our signal trampoline
 *	as the signal handler instead.  The code here is derived
 *	from sigvec in sys/kern_sig.c.
 */
extern int __sigaction (int, struct __sigaction * __restrict,
		struct sigaction * __restrict);

int
__platform_sigaction (int sig, const struct sigaction * __restrict nsv,
		struct sigaction * __restrict osv)
{
	extern void _sigtramp();
	struct __sigaction sa;
	struct __sigaction *sap;
	int ret;

	if (sig <= 0 || sig >= NSIG || sig == SIGKILL || sig == SIGSTOP) {
	        errno = EINVAL;
	        return (-1);
	}
	sap = (struct __sigaction *)0;
	if (nsv) {
		sa.sa_handler = nsv->sa_handler;
		sa.sa_tramp = _sigtramp;
		sa.sa_mask = nsv->sa_mask;
		sa.sa_flags = nsv->sa_flags | SA_VALIDATE_SIGRETURN_FROM_SIGTRAMP;
		sap = &sa;
	}
	ret = __sigaction(sig, sap, osv);
	return ret;
}
