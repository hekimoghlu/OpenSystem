/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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
#ifndef _SYS_SYSENT_H_
#define _SYS_SYSENT_H_

#include <sys/appleapiopts.h>
#include <sys/cdefs.h>

#ifdef KERNEL_PRIVATE
#ifdef __APPLE_API_PRIVATE

typedef int32_t sy_call_t(struct proc *, void *, int *);
#if CONFIG_REQUIRES_U32_MUNGING
typedef void    sy_munge_t(void *);
#endif

struct sysent {         /* system call table */
	sy_call_t       *sy_call;       /* implementing function */
#if CONFIG_REQUIRES_U32_MUNGING
	sy_munge_t      *sy_arg_munge32; /* system call arguments munger for 32-bit process */
#endif
	int32_t         sy_return_type; /* system call return types */
	int16_t         sy_narg;        /* number of args */
	uint16_t        sy_arg_bytes;   /* Total size of arguments in bytes for
	                                 * 32-bit system calls
	                                 */
};

extern const struct sysent sysent[];
extern const unsigned int nsysent;

/*
 * Valid values for sy_cancel
 */
#define _SYSCALL_CANCEL_NONE    0               /* Not a cancellation point */
#define _SYSCALL_CANCEL_PRE             1               /* Canbe cancelled on entry itself */
#define _SYSCALL_CANCEL_POST    2               /* Can only be cancelled after syscall is run */

/*
 * Valid values for sy_return_type
 */
#define _SYSCALL_RET_NONE               0
#define _SYSCALL_RET_INT_T              1
#define _SYSCALL_RET_UINT_T             2
#define _SYSCALL_RET_OFF_T              3
#define _SYSCALL_RET_ADDR_T             4
#define _SYSCALL_RET_SIZE_T             5
#define _SYSCALL_RET_SSIZE_T    6
#define _SYSCALL_RET_UINT64_T   7

#endif /* __APPLE_API_PRIVATE */
#endif /* KERNEL_PRIVATE */

#endif /* !_SYS_SYSENT_H_ */
