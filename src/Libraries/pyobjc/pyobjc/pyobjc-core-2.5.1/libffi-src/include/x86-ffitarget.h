/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 27, 2024.
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

#if defined(X86_64) && defined(__i386__)
#	undef X86_64
#	define X86
#endif

#if defined(__x86_64__) 
#	ifndef X86_64
#		define X86_64
#	endif
#endif

/* ---- Generic type definitions ----------------------------------------- */

#ifndef LIBFFI_ASM

typedef unsigned long	ffi_arg;
typedef signed long		ffi_sarg;

typedef enum ffi_abi {
	FFI_FIRST_ABI = 0,

	/* ---- Intel x86 Win32 ---------- */
#ifdef X86_WIN32
	FFI_SYSV,
	FFI_STDCALL,
	/* TODO: Add fastcall support for the sake of completeness */
	FFI_DEFAULT_ABI = FFI_SYSV,
#endif

	/* ---- Intel x86 and AMD x86-64 - */
#if !defined(X86_WIN32) && (defined(__i386__) || defined(__x86_64__))
	FFI_SYSV,
	FFI_UNIX64,   /* Unix variants all use the same ABI for x86-64  */
#	ifdef __i386__
	FFI_DEFAULT_ABI = FFI_SYSV,
#	else
	FFI_DEFAULT_ABI = FFI_UNIX64,
#	endif
#endif

	FFI_LAST_ABI = FFI_DEFAULT_ABI + 1
} ffi_abi;

#endif	// #ifndef LIBFFI_ASM

/* ---- Definitions for closures ----------------------------------------- */

#define FFI_CLOSURES 1

#if defined(X86_64) || (defined(__x86_64__) && defined(X86_DARWIN))
#	define FFI_TRAMPOLINE_SIZE 24
#	define FFI_NATIVE_RAW_API 0
#else
#	define FFI_TRAMPOLINE_SIZE 10
#	define FFI_NATIVE_RAW_API 1	/* x86 has native raw api support */
#endif

#endif	// #ifndef LIBFFI_TARGET_H