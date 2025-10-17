/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 18, 2023.
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
 * Mach Operating System
 * Copyright (c) 1991,1990,1989 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS"
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie Mellon
 * the rights to redistribute these changes.
 */

/*
 * Machine dependent task fields
 */

#ifdef MACH_KERNEL_PRIVATE
/* Provide access to target-specific defintions which may be used by
 * consuming code, e.g. HYPERVISOR. */
#include <arm64/proc_reg.h>
#endif


#if defined(HAS_APPLE_PAC)
#define TASK_ADDITIONS_PAC \
	uint64_t rop_pid; \
	uint64_t jop_pid; \
	uint8_t disable_user_jop;
#else
#define TASK_ADDITIONS_PAC
#endif



#define TASK_ADDITIONS_UEXC uint64_t uexc[4];

#if !__ARM_KERNEL_PROTECT__
#define TASK_ADDITIONS_X18 bool preserve_x18;
#else
#define TASK_ADDITIONS_X18
#endif

#define TASK_ADDITIONS_APT

#define MACHINE_TASK \
	void * XNU_PTRAUTH_SIGNED_PTR("task.task_debug") task_debug; \
	TASK_ADDITIONS_PAC \
\
	TASK_ADDITIONS_UEXC \
	TASK_ADDITIONS_X18 \
	TASK_ADDITIONS_APT \
	bool uses_1ghz_timebase;
