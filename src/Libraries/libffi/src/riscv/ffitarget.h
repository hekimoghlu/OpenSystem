/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 26, 2024.
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

#ifndef __riscv
#error "libffi was configured for a RISC-V target but this does not appear to be a RISC-V compiler."
#endif

#ifndef LIBFFI_ASM

typedef unsigned long ffi_arg;
typedef   signed long ffi_sarg;

/* FFI_UNUSED_NN and riscv_unused are to maintain ABI compatibility with a
   distributed Berkeley patch from 2014, and can be removed at SONAME bump */
typedef enum ffi_abi {
    FFI_FIRST_ABI = 0,
    FFI_SYSV,
    FFI_UNUSED_1,
    FFI_UNUSED_2,
    FFI_UNUSED_3,
    FFI_LAST_ABI,

    FFI_DEFAULT_ABI = FFI_SYSV
} ffi_abi;

#endif /* LIBFFI_ASM */

/* ---- Definitions for closures ----------------------------------------- */

#define FFI_CLOSURES 1
#define FFI_GO_CLOSURES 1
#define FFI_TRAMPOLINE_SIZE 24
#define FFI_NATIVE_RAW_API 0
#define FFI_EXTRA_CIF_FIELDS unsigned riscv_nfixedargs; unsigned riscv_unused;
#define FFI_TARGET_SPECIFIC_VARIADIC

#endif

