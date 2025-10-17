/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 12, 2024.
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

#ifndef LIBFFI_ASM
typedef unsigned long          ffi_arg;
typedef signed long            ffi_sarg;

typedef enum ffi_abi {
  FFI_FIRST_ABI = 0,

#ifdef PA_LINUX
  FFI_PA32,
  FFI_LAST_ABI,
  FFI_DEFAULT_ABI = FFI_PA32
#endif

#ifdef PA_HPUX
  FFI_PA32,
  FFI_LAST_ABI,
  FFI_DEFAULT_ABI = FFI_PA32
#endif

#ifdef PA64_HPUX
#error "PA64_HPUX FFI is not yet implemented"
  FFI_PA64,
  FFI_LAST_ABI,
  FFI_DEFAULT_ABI = FFI_PA64
#endif
} ffi_abi;
#endif

#define FFI_TARGET_SPECIFIC_STACK_SPACE_ALLOCATION

/* ---- Definitions for closures ----------------------------------------- */

#define FFI_CLOSURES 1
#define FFI_NATIVE_RAW_API 0

#ifdef PA_LINUX
#define FFI_TRAMPOLINE_SIZE 32
#else
#define FFI_TRAMPOLINE_SIZE 40
#endif

#define FFI_TYPE_SMALL_STRUCT2 -1
#define FFI_TYPE_SMALL_STRUCT3 -2
#define FFI_TYPE_SMALL_STRUCT4 -3
#define FFI_TYPE_SMALL_STRUCT5 -4
#define FFI_TYPE_SMALL_STRUCT6 -5
#define FFI_TYPE_SMALL_STRUCT7 -6
#define FFI_TYPE_SMALL_STRUCT8 -7
#endif
