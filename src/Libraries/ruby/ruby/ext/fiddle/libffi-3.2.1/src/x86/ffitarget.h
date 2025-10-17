/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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

#ifndef LIBFFI_H
#error "Please do not include ffitarget.h directly into your source.  Use ffi.h instead."
#endif

/* ---- System specific configurations ----------------------------------- */

/* For code common to all platforms on x86 and x86_64. */
#define X86_ANY

#if defined (X86_64) && defined (__i386__)
#undef X86_64
#define X86
#endif

#ifdef X86_WIN64
#define FFI_SIZEOF_ARG 8
#define USE_BUILTIN_FFS 0 /* not yet implemented in mingw-64 */
#endif

#define FFI_TARGET_SPECIFIC_STACK_SPACE_ALLOCATION
#ifndef _MSC_VER
#define FFI_TARGET_HAS_COMPLEX_TYPE
#endif

/* ---- Generic type definitions ----------------------------------------- */

#ifndef LIBFFI_ASM
#ifdef X86_WIN64
#ifdef _MSC_VER
typedef unsigned __int64       ffi_arg;
typedef __int64                ffi_sarg;
#else
typedef unsigned long long     ffi_arg;
typedef long long              ffi_sarg;
#endif
#else
#if defined __x86_64__ && defined __ILP32__
#define FFI_SIZEOF_ARG 8
#define FFI_SIZEOF_JAVA_RAW  4
typedef unsigned long long     ffi_arg;
typedef long long              ffi_sarg;
#else
typedef unsigned long          ffi_arg;
typedef signed long            ffi_sarg;
#endif
#endif

typedef enum ffi_abi {
  FFI_FIRST_ABI = 0,

  /* ---- Intel x86 Win32 ---------- */
#ifdef X86_WIN32
  FFI_SYSV,
  FFI_STDCALL,
  FFI_THISCALL,
  FFI_FASTCALL,
  FFI_MS_CDECL,
  FFI_PASCAL,
  FFI_REGISTER,
  FFI_LAST_ABI,
#ifdef _MSC_VER
  FFI_DEFAULT_ABI = FFI_MS_CDECL
#else
  FFI_DEFAULT_ABI = FFI_SYSV
#endif

#elif defined(X86_WIN64)
  FFI_WIN64,
  FFI_LAST_ABI,
  FFI_DEFAULT_ABI = FFI_WIN64

#else
  /* ---- Intel x86 and AMD x86-64 - */
  FFI_SYSV,
  FFI_UNIX64,   /* Unix variants all use the same ABI for x86-64  */
  FFI_THISCALL,
  FFI_FASTCALL,
  FFI_STDCALL,
  FFI_PASCAL,
  FFI_REGISTER,
  FFI_LAST_ABI,
#if defined(__i386__) || defined(__i386)
  FFI_DEFAULT_ABI = FFI_SYSV
#else
  FFI_DEFAULT_ABI = FFI_UNIX64
#endif
#endif
} ffi_abi;
#endif

/* ---- Definitions for closures ----------------------------------------- */

#define FFI_CLOSURES 1
#define FFI_TYPE_SMALL_STRUCT_1B (FFI_TYPE_LAST + 1)
#define FFI_TYPE_SMALL_STRUCT_2B (FFI_TYPE_LAST + 2)
#define FFI_TYPE_SMALL_STRUCT_4B (FFI_TYPE_LAST + 3)
#define FFI_TYPE_MS_STRUCT       (FFI_TYPE_LAST + 4)

#if defined (X86_64) || (defined (__x86_64__) && defined (X86_DARWIN))
#define FFI_TRAMPOLINE_SIZE 24
#define FFI_NATIVE_RAW_API 0
#else
#ifdef X86_WIN32
#define FFI_TRAMPOLINE_SIZE 52
#else
#ifdef X86_WIN64
#define FFI_TRAMPOLINE_SIZE 29
#define FFI_NATIVE_RAW_API 0
#define FFI_NO_RAW_API 1
#else
#define FFI_TRAMPOLINE_SIZE 10
#endif
#endif
#ifndef X86_WIN64
#define FFI_NATIVE_RAW_API 1  /* x86 has native raw api support */
#endif
#endif

#endif

