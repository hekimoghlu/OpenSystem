/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 21, 2024.
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
 * Mach Operating System
 * Copyright (c) 1987 Carnegie-Mellon University
 * All rights reserved.  The CMU software License Agreement specifies
 * the terms and conditions for use and redistribution.
 */

#include <sys/param.h>

#include <mach/boolean.h>
#include <mach/exception.h>
#include <mach/kern_return.h>

#include <sys/proc.h>
#include <sys/user.h>
#include <sys/systm.h>
#include <sys/vmparam.h>        /* MAXSSIZ */

#include <sys/ux_exception.h>

/*
 * Translate Mach exceptions to UNIX signals.
 *
 * ux_exception translates a mach exception, code and subcode to
 * a signal.  Calls machine_exception (machine dependent)
 * to attempt translation first.
 */
static int
ux_exception(int                        exception,
    mach_exception_code_t      code,
    mach_exception_subcode_t   subcode)
{
	int machine_signal = 0;

	/* Try machine-dependent translation first. */
	if ((machine_signal = machine_exception(exception, code, subcode)) != 0) {
		return machine_signal;
	}

	switch (exception) {
	case EXC_BAD_ACCESS:
		if (code == KERN_INVALID_ADDRESS) {
			return SIGSEGV;
		} else {
			return SIGBUS;
		}

	case EXC_SYSCALL:
		if (send_sigsys) {
			return SIGSYS;
		}
		break;

	case EXC_BAD_INSTRUCTION:
		return SIGILL;

	case EXC_ARITHMETIC:
		return SIGFPE;

	case EXC_EMULATION:
		return SIGEMT;

	case EXC_SOFTWARE:
		switch (code) {
		case EXC_UNIX_BAD_SYSCALL:
			return SIGSYS;
		case EXC_UNIX_BAD_PIPE:
			return SIGPIPE;
		case EXC_UNIX_ABORT:
			return SIGABRT;
		case EXC_SOFT_SIGNAL:
			return SIGKILL;
		}
		break;

	case EXC_BREAKPOINT:
		return SIGTRAP;
	}

	return 0;
}

/*
 * Sends the corresponding UNIX signal to a thread that has triggered a Mach exception.
 */
kern_return_t
handle_ux_exception(thread_t                    thread,
    int                         exception,
    mach_exception_code_t       code,
    mach_exception_subcode_t    subcode)
{
	/* Returns +1 proc reference */
	proc_t p = proc_findthread(thread);

	/* Can't deliver a signal without a bsd process reference */
	if (p == NULL) {
		return KERN_FAILURE;
	}

	/* Translate exception and code to signal type */
	int ux_signal = ux_exception(exception, code, subcode);

	uthread_t ut = get_bsdthread_info(thread);

	/*
	 * Stack overflow should result in a SIGSEGV signal
	 * on the alternate stack.
	 * but we have one or more guard pages after the
	 * stack top, so we would get a KERN_PROTECTION_FAILURE
	 * exception instead of KERN_INVALID_ADDRESS, resulting in
	 * a SIGBUS signal.
	 * Detect that situation and select the correct signal.
	 */
	if (code == KERN_PROTECTION_FAILURE &&
	    ux_signal == SIGBUS) {
		user_addr_t sp = subcode;

		user_addr_t stack_max = p->user_stack;
		user_addr_t stack_min = p->user_stack - MAXSSIZ;
		if (sp >= stack_min && sp < stack_max) {
			/*
			 * This is indeed a stack overflow.  Deliver a
			 * SIGSEGV signal.
			 */
			ux_signal = SIGSEGV;

			/*
			 * If the thread/process is not ready to handle
			 * SIGSEGV on an alternate stack, force-deliver
			 * SIGSEGV with a SIG_DFL handler.
			 */
			int mask = sigmask(ux_signal);
			struct sigacts *ps = &p->p_sigacts;
			if ((p->p_sigignore & mask) ||
			    (ut->uu_sigwait & mask) ||
			    (ut->uu_sigmask & mask) ||
			    (SIGACTION(p, SIGSEGV) == SIG_IGN) ||
			    (!(ps->ps_sigonstack & mask))) {
				p->p_sigignore &= ~mask;
				p->p_sigcatch &= ~mask;
				proc_set_sigact(p, SIGSEGV, SIG_DFL);
				ut->uu_sigwait &= ~mask;
				ut->uu_sigmask &= ~mask;
			}
		}
	}

	/* Send signal to thread */
	if (ux_signal != 0) {
		ut->uu_exception = exception;
		//ut->uu_code = code; // filled in by threadsignal
		ut->uu_subcode = subcode;
		threadsignal(thread, ux_signal, code, TRUE);
	}

	proc_rele(p);

	return KERN_SUCCESS;
}

