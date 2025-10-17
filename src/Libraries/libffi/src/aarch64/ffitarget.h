/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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

#ifndef LIBFFI_ASM
#ifdef __ILP32__
#define FFI_SIZEOF_ARG 8
#define FFI_SIZEOF_JAVA_RAW  4
typedef unsigned long long ffi_arg;
typedef signed long long ffi_sarg;
#elif defined(_WIN32)
#define FFI_SIZEOF_ARG 8
typedef unsigned long long ffi_arg;
typedef signed long long ffi_sarg;
#else
typedef unsigned long ffi_arg;
typedef signed long ffi_sarg;
#endif

typedef enum ffi_abi
  {
    FFI_FIRST_ABI = 0,
    FFI_SYSV,
    FFI_WIN64,
    FFI_LAST_ABI,
#if defined(_WIN32)
    FFI_DEFAULT_ABI = FFI_WIN64
#else
    FFI_DEFAULT_ABI = FFI_SYSV
#endif
  } ffi_abi;
#endif

/* ---- Definitions for closures ----------------------------------------- */

#define FFI_CLOSURES 1
#define FFI_NATIVE_RAW_API 0

#if defined (FFI_EXEC_TRAMPOLINE_TABLE) && FFI_EXEC_TRAMPOLINE_TABLE

#ifdef __MACH__
#define FFI_TRAMPOLINE_SIZE 16
#define FFI_TRAMPOLINE_CLOSURE_OFFSET 16
#else
#error "No trampoline table implementation"
#endif

#else
#define FFI_TRAMPOLINE_SIZE 24
#define FFI_TRAMPOLINE_CLOSURE_OFFSET FFI_TRAMPOLINE_SIZE
#endif

#ifdef _WIN32
#define FFI_EXTRA_CIF_FIELDS unsigned is_variadic
#endif
#define FFI_TARGET_SPECIFIC_VARIADIC

/* ---- Internal ---- */

#if defined (__APPLE__)
#define FFI_EXTRA_CIF_FIELDS unsigned aarch64_nfixedargs
#elif !defined(_WIN32)
/* iOS and Windows reserve x18 for the system.  Disable Go closures until
   a new static chain is chosen.  */
#define FFI_GO_CLOSURES 1
#endif

#ifndef _WIN32
/* No complex type on Windows */
#define FFI_TARGET_HAS_COMPLEX_TYPE
#endif

#endif
