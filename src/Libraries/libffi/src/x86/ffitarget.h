/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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
#if defined(X86_WIN64)
  FFI_FIRST_ABI = 0,
  FFI_WIN64,            /* sizeof(long double) == 8  - microsoft compilers */
  FFI_GNUW64,           /* sizeof(long double) == 16 - GNU compilers */
  FFI_LAST_ABI,
#ifdef __GNUC__
  FFI_DEFAULT_ABI = FFI_GNUW64
#else  
  FFI_DEFAULT_ABI = FFI_WIN64
#endif  

#elif defined(X86_64) || (defined (__x86_64__) && defined (X86_DARWIN))
  FFI_FIRST_ABI = 1,
  FFI_UNIX64,
  FFI_WIN64,
  FFI_EFI64 = FFI_WIN64,
  FFI_GNUW64,
  FFI_LAST_ABI,
  FFI_DEFAULT_ABI = FFI_UNIX64

#elif defined(X86_WIN32)
  FFI_FIRST_ABI = 0,
  FFI_SYSV      = 1,
  FFI_STDCALL   = 2,
  FFI_THISCALL  = 3,
  FFI_FASTCALL  = 4,
  FFI_MS_CDECL  = 5,
  FFI_PASCAL    = 6,
  FFI_REGISTER  = 7,
  FFI_LAST_ABI,
  FFI_DEFAULT_ABI = FFI_MS_CDECL
#else
  FFI_FIRST_ABI = 0,
  FFI_SYSV      = 1,
  FFI_THISCALL  = 3,
  FFI_FASTCALL  = 4,
  FFI_STDCALL   = 5,
  FFI_PASCAL    = 6,
  FFI_REGISTER  = 7,
  FFI_MS_CDECL  = 8,
  FFI_LAST_ABI,
  FFI_DEFAULT_ABI = FFI_SYSV
#endif
} ffi_abi;
#endif

/* ---- Definitions for closures ----------------------------------------- */

#define FFI_CLOSURES 1
#define FFI_GO_CLOSURES 1

#define FFI_TYPE_SMALL_STRUCT_1B (FFI_TYPE_LAST + 1)
#define FFI_TYPE_SMALL_STRUCT_2B (FFI_TYPE_LAST + 2)
#define FFI_TYPE_SMALL_STRUCT_4B (FFI_TYPE_LAST + 3)
#define FFI_TYPE_MS_STRUCT       (FFI_TYPE_LAST + 4)

#if defined (X86_64) || defined(X86_WIN64) \
    || (defined (__x86_64__) && defined (X86_DARWIN))
# define FFI_TRAMPOLINE_SIZE 24
# define FFI_NATIVE_RAW_API 0
#else
# define FFI_TRAMPOLINE_SIZE 12
# define FFI_NATIVE_RAW_API 1  /* x86 has native raw api support */
#endif

#endif

