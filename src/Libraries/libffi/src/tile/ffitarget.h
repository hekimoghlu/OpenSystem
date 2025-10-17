/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 9, 2024.
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

#include <arch/abi.h>

typedef uint_reg_t ffi_arg;
typedef int_reg_t  ffi_sarg;

typedef enum ffi_abi {
  FFI_FIRST_ABI = 0,
  FFI_UNIX,
  FFI_LAST_ABI,
  FFI_DEFAULT_ABI = FFI_UNIX
} ffi_abi;
#endif

/* ---- Definitions for closures ----------------------------------------- */
#define FFI_CLOSURES 1

#ifdef __tilegx__
/* We always pass 8-byte values, even in -m32 mode. */
# define FFI_SIZEOF_ARG 8
# ifdef __LP64__
#  define FFI_TRAMPOLINE_SIZE (8 * 5)  /* 5 bundles */
# else
#  define FFI_TRAMPOLINE_SIZE (8 * 3)  /* 3 bundles */
# endif
#else
# define FFI_SIZEOF_ARG 4
# define FFI_TRAMPOLINE_SIZE 8 /* 1 bundle */
#endif
#define FFI_NATIVE_RAW_API 0

#endif
