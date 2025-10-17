/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 31, 2022.
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
 * Copyright (c) 1989 Carnegie-Mellon University
 * Copyright (c) 1988 Carnegie-Mellon University
 * Copyright (c) 1987 Carnegie-Mellon University
 * All rights reserved.  The CMU software License Agreement specifies
 * the terms and conditions for use and redistribution.
 */

/*
 *	Codes for Unix software exceptions under EXC_SOFTWARE.
 */

#ifndef _SYS_UX_EXCEPTION_H_
#define _SYS_UX_EXCEPTION_H_

#include <sys/appleapiopts.h>

#ifdef __APPLE_API_UNSTABLE

#define EXC_UNIX_BAD_SYSCALL    0x10000         /* SIGSYS */

#define EXC_UNIX_BAD_PIPE       0x10001         /* SIGPIPE */

#define EXC_UNIX_ABORT          0x10002         /* SIGABRT */

#endif /* __APPLE_API_UNSTABLE */

#ifdef XNU_KERNEL_PRIVATE

/* Kernel functions for Unix exception handler. */

#include <mach/mach_types.h>

extern int
machine_exception(int exception, mach_exception_code_t code,
    mach_exception_subcode_t subcode);

extern kern_return_t
handle_ux_exception(thread_t thread, int exception,
    mach_exception_code_t code,
    mach_exception_subcode_t subcode);

#endif /* XNU_KERNEL_PRIVATE */

#endif  /* _SYS_UX_EXCEPTION_H_ */
