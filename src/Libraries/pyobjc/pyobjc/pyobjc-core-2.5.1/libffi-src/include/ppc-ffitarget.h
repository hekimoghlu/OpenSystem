/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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
#ifndef LIBFFI_TARGET_H
#define LIBFFI_TARGET_H

/* ---- System specific configurations ----------------------------------- */

#if (defined(POWERPC) && defined(__powerpc64__)) ||		\
	(defined(POWERPC_DARWIN) && defined(__ppc64__))
#define POWERPC64
#endif

#ifndef LIBFFI_ASM

typedef unsigned long	ffi_arg;
typedef signed long		ffi_sarg;

typedef enum ffi_abi {
	FFI_FIRST_ABI = 0,

#ifdef POWERPC
	FFI_SYSV,
	FFI_GCC_SYSV,
	FFI_LINUX64,
#	ifdef POWERPC64
	FFI_DEFAULT_ABI = FFI_LINUX64,
#	else
	FFI_DEFAULT_ABI = FFI_GCC_SYSV,
#	endif
#endif

#ifdef POWERPC_AIX
	FFI_AIX,
	FFI_DARWIN,
	FFI_DEFAULT_ABI = FFI_AIX,
#endif

#ifdef POWERPC_DARWIN
	FFI_AIX,
	FFI_DARWIN,
	FFI_DEFAULT_ABI = FFI_DARWIN,
#endif

#ifdef POWERPC_FREEBSD
	FFI_SYSV,
	FFI_GCC_SYSV,
	FFI_LINUX64,
	FFI_DEFAULT_ABI = FFI_SYSV,
#endif

	FFI_LAST_ABI = FFI_DEFAULT_ABI + 1
} ffi_abi;

#endif	// #ifndef LIBFFI_ASM

/* ---- Definitions for closures ----------------------------------------- */

#define FFI_CLOSURES 1
#define FFI_NATIVE_RAW_API 0

/* Needed for FFI_SYSV small structure returns.  */
#define FFI_SYSV_TYPE_SMALL_STRUCT  (FFI_TYPE_LAST)

#if defined(POWERPC64) /*|| defined(POWERPC_AIX)*/
#	define FFI_TRAMPOLINE_SIZE 48
#elif defined(POWERPC_AIX)
#	define FFI_TRAMPOLINE_SIZE 24
#else
#	define FFI_TRAMPOLINE_SIZE 40
#endif

#ifndef LIBFFI_ASM
#	if defined(POWERPC_DARWIN) || defined(POWERPC_AIX)
typedef struct ffi_aix_trampoline_struct {
	void*	code_pointer;	/* Pointer to ffi_closure_ASM */
	void*	toc;			/* TOC */
	void*	static_chain;	/* Pointer to closure */
} ffi_aix_trampoline_struct;
#	endif
#endif	// #ifndef LIBFFI_ASM

#endif	// #ifndef LIBFFI_TARGET_H