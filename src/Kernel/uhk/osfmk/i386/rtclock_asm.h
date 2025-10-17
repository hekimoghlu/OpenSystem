/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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
 * @OSF_COPYRIGHT@
 */
/*
 * @APPLE_FREE_COPYRIGHT@
 */
/*
 *	File:		rtclock_asm.h
 *	Purpose:	Assembly routines for handling the machine dependent
 *				real-time clock.
 */

#ifndef _I386_RTCLOCK_H_
#define _I386_RTCLOCK_H_

#include <i386/pal_rtclock_asm.h>

/*
 * Update time on user trap entry.
 * Uses: %rsi, %rdi, %rdx, %rcx, %rax
 */
#define	TIME_TRAP_UENTRY CCALL(recount_leave_user)

/*
 * update time on user trap exit.
 * Uses: %rsi, %rdi, %rdx, %rcx, %rax
 */
#define	TIME_TRAP_UEXIT CCALL(recount_enter_user)

/*
 * Nanotime returned in %rax.
 * Computed from tsc based on the scale factor and an implicit 32 bit shift.
 * This code must match what _rtc_nanotime_read does in
 * machine_routines_asm.s.  Failure to do so can
 * result in "weird" timing results.
 *
 * Uses: %rsi, %rdi, %rdx, %rcx
 */
#define NANOTIME							       \
	movq	%gs:CPU_NANOTIME,%rdi					     ; \
	PAL_RTC_NANOTIME_READ_FAST()

/*
 * Check for vtimers for task.
 *   task_reg   is register pointing to current task
 *   thread_reg is register pointing to current thread
 */
#define TASK_VTIMER_CHECK(task_reg,thread_reg)				       \
	cmpl	$0,TASK_VTIMERS(task_reg)				     ; \
	jz	1f							     ; \
	orl	$(AST_BSD),%gs:CPU_PENDING_AST	/* Set pending AST */	     ; \
	lock								     ; \
	orl	$(AST_BSD),TH_AST(thread_reg)	/* Set thread AST  */	     ; \
1:									     ; \

#endif /* _I386_RTCLOCK_H_ */
