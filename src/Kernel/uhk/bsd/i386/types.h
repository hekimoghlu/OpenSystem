/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 26, 2025.
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
 * Copyright 1995 NeXT Computer, Inc. All rights reserved.
 */
/*
 * Copyright (c) 1990, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 *	@(#)types.h	8.3 (Berkeley) 1/5/94
 */

#ifndef _I386_MACHTYPES_H_
#define _I386_MACHTYPES_H_
#define _MACHTYPES_H_

#if defined (__i386__) || defined (__x86_64__)

#ifndef __ASSEMBLER__
#include <i386/_types.h>
#include <sys/cdefs.h>
/*
 * Basic integral types.  Omit the typedef if
 * not possible for a machine/compiler combination.
 */
#include <sys/_types/_int8_t.h>
#include <sys/_types/_int16_t.h>
#include <sys/_types/_int32_t.h>
#include <sys/_types/_int64_t.h>

#include <sys/_types/_u_int8_t.h>
#include <sys/_types/_u_int16_t.h>
#include <sys/_types/_u_int32_t.h>
#include <sys/_types/_u_int64_t.h>

#if __LP64__
typedef int64_t                 register_t;
#else
typedef int32_t                 register_t;
#endif

#include <sys/_types/_intptr_t.h>
#include <sys/_types/_uintptr_t.h>

#if !defined(_ANSI_SOURCE) && (!defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE))
/* These types are used for reserving the largest possible size. */
typedef u_int64_t               user_addr_t;
typedef u_int64_t               user_size_t;
typedef int64_t                 user_ssize_t;
typedef int64_t                 user_long_t;
typedef u_int64_t               user_ulong_t;
typedef int64_t                 user_time_t;
typedef int64_t                 user_off_t;

#if KERNEL
#ifndef VM_UNSAFE_TYPES
typedef user_addr_t             user_addr_ut;
typedef user_size_t             user_size_ut;
#endif /* VM_SAFE_TYPES */
#endif /* KERNEL */

#define USER_ADDR_NULL  ((user_addr_t) 0)
#define CAST_USER_ADDR_T(a_ptr)   ((user_addr_t)((uintptr_t)(a_ptr)))

#ifdef KERNEL

/*
 * These types are used when you know the word size of the target
 * user process. They can be used to create struct layouts independent
 * of the types and alignment requirements of the current running
 * kernel.
 */

/*
 * The default ABI for the 32-bit Intel userspace aligns fundamental
 * integral data types to their natural boundaries, with a maximum alignment
 * of 4, even for 8-byte quantites. The default ABI for 64-bit Intel
 * userspace aligns fundamental integral data types for their natural
 * boundaries, including those in composite data types. PowerPC applications
 * running under translation must conform to the 32-bit Intel ABI.
 */

typedef __uint64_t              user64_addr_t __attribute__((aligned(8)));
typedef __uint64_t              user64_size_t __attribute__((aligned(8)));
typedef __int64_t               user64_ssize_t __attribute__((aligned(8)));
typedef __int64_t               user64_long_t __attribute__((aligned(8)));
typedef __uint64_t              user64_ulong_t __attribute__((aligned(8)));
typedef __int64_t               user64_time_t __attribute__((aligned(8)));
typedef __int64_t               user64_off_t __attribute__((aligned(8)));

typedef __uint32_t              user32_addr_t;
typedef __uint32_t              user32_size_t;
typedef __int32_t               user32_ssize_t;
typedef __int32_t               user32_long_t;
typedef __uint32_t              user32_ulong_t;
typedef __int32_t               user32_time_t;
typedef __int64_t               user32_off_t __attribute__((aligned(4)));

#endif /* KERNEL */

#endif /* !_ANSI_SOURCE && (!_POSIX_C_SOURCE || _DARWIN_C_SOURCE) */

/* This defines the size of syscall arguments after copying into the kernel: */
typedef u_int64_t               syscall_arg_t;

#endif /* __ASSEMBLER__ */
#endif /* defined (__i386__) || defined (__x86_64__) */
#endif  /* _I386_MACHTYPES_H_ */
