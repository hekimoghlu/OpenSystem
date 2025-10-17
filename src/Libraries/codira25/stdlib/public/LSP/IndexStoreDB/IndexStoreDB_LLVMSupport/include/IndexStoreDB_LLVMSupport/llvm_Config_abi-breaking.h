/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://toolchain.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

/* This file controls the C++ ABI break introduced in LLVM public header. */

#ifndef LLVM_ABI_BREAKING_CHECKS_H
#define LLVM_ABI_BREAKING_CHECKS_H

#include <IndexStoreDB_LLVMSupport/toolchain_Config_indexstoredb-prefix.h>

/* Define to enable checks that alter the LLVM C++ ABI */
#define LLVM_ENABLE_ABI_BREAKING_CHECKS 0

/* Define to enable reverse iteration of unordered toolchain containers */
#define LLVM_ENABLE_REVERSE_ITERATION 0

/* Allow selectively disabling link-time mismatch checking so that header-only
   ADT content from LLVM can be used without linking libSupport. */
#if !LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING

// ABI_BREAKING_CHECKS protection: provides link-time failure when clients build
// mismatch with LLVM
#if defined(_MSC_VER)
// Use pragma with MSVC
#define LLVM_XSTR(s) LLVM_STR(s)
#define LLVM_STR(s) #s
#pragma detect_mismatch("LLVM_ENABLE_ABI_BREAKING_CHECKS", LLVM_XSTR(LLVM_ENABLE_ABI_BREAKING_CHECKS))
#undef LLVM_XSTR
#undef LLVM_STR
#elif defined(_WIN32) || defined(__CYGWIN__) // Win32 w/o #pragma detect_mismatch
// FIXME: Implement checks without weak.
#elif defined(__cplusplus)
#if !(defined(_AIX) && defined(__GNUC__) && !defined(__clang__))
#define LLVM_HIDDEN_VISIBILITY __attribute__ ((visibility("hidden")))
#else
// GCC on AIX does not support visibility attributes. Symbols are not
// exported by default on AIX.
#define LLVM_HIDDEN_VISIBILITY
#endif
namespace toolchain {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
extern int EnableABIBreakingChecks;
LLVM_HIDDEN_VISIBILITY
__attribute__((weak)) int *VerifyEnableABIBreakingChecks =
    &EnableABIBreakingChecks;
#else
extern int DisableABIBreakingChecks;
LLVM_HIDDEN_VISIBILITY
__attribute__((weak)) int *VerifyDisableABIBreakingChecks =
    &DisableABIBreakingChecks;
#endif
}
#undef LLVM_HIDDEN_VISIBILITY
#endif // _MSC_VER

#endif // LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING

#endif
