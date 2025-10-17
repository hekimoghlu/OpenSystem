/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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

//===- llvm/Support/FEnv.h - Host floating-point exceptions ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides an operating system independent interface to
// floating-point exception interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_FENV_H
#define LLVM_SYSTEM_FENV_H

#include "llvm/Config/config.h"
#include <cerrno>
#ifdef HAVE_FENV_H
#include <fenv.h>
#endif

// FIXME: Clang's #include handling apparently doesn't work for libstdc++'s
// fenv.h; see PR6907 for details.
#if defined(__clang__) && defined(_GLIBCXX_FENV_H)
#undef HAVE_FENV_H
#endif

namespace llvm {
namespace sys {

/// llvm_fenv_clearexcept - Clear the floating-point exception state.
static inline void llvm_fenv_clearexcept() {
#ifdef HAVE_FENV_H
  feclearexcept(FE_ALL_EXCEPT);
#endif
  errno = 0;
}

/// llvm_fenv_testexcept - Test if a floating-point exception was raised.
static inline bool llvm_fenv_testexcept() {
  int errno_val = errno;
  if (errno_val == ERANGE || errno_val == EDOM)
    return true;
#ifdef HAVE_FENV_H
  if (fetestexcept(FE_ALL_EXCEPT & ~FE_INEXACT))
    return true;
#endif
  return false;
}

} // End sys namespace
} // End llvm namespace

#endif
