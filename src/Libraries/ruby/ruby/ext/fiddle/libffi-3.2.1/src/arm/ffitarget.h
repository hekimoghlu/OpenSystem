/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 5, 2024.
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
typedef unsigned long          ffi_arg;
typedef signed long            ffi_sarg;

typedef enum ffi_abi {
  FFI_FIRST_ABI = 0,
  FFI_SYSV,
  FFI_VFP,
  FFI_LAST_ABI,
#ifdef __ARM_PCS_VFP
  FFI_DEFAULT_ABI = FFI_VFP,
#else
  FFI_DEFAULT_ABI = FFI_SYSV,
#endif
} ffi_abi;
#endif

#define FFI_EXTRA_CIF_FIELDS			\
  int vfp_used;					\
  short vfp_reg_free, vfp_nargs;		\
  signed char vfp_args[16]			\

/* Internally used. */
#define FFI_TYPE_STRUCT_VFP_FLOAT  (FFI_TYPE_LAST + 1)
#define FFI_TYPE_STRUCT_VFP_DOUBLE (FFI_TYPE_LAST + 2)

#define FFI_TARGET_SPECIFIC_VARIADIC

/* ---- Definitions for closures ----------------------------------------- */

#define FFI_CLOSURES 1
#define FFI_TRAMPOLINE_SIZE 20
#define FFI_NATIVE_RAW_API 0

#endif
