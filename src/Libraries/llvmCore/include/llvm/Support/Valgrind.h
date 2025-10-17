/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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

//===- llvm/Support/Valgrind.h - Communication with Valgrind -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Methods for communicating with a valgrind instance this program is running
// under.  These are all no-ops unless LLVM was configured on a system with the
// valgrind headers installed and valgrind is controlling this process.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_VALGRIND_H
#define LLVM_SYSTEM_VALGRIND_H

#include "llvm/Support/Compiler.h"
#include "llvm/Config/llvm-config.h"
#include <stddef.h>

#if LLVM_ENABLE_THREADS != 0 && !defined(NDEBUG)
// tsan (Thread Sanitizer) is a valgrind-based tool that detects these exact
// functions by name.
extern "C" {
LLVM_ATTRIBUTE_WEAK void AnnotateHappensAfter(const char *file, int line,
                                              const volatile void *cv);
LLVM_ATTRIBUTE_WEAK void AnnotateHappensBefore(const char *file, int line,
                                               const volatile void *cv);
LLVM_ATTRIBUTE_WEAK void AnnotateIgnoreWritesBegin(const char *file, int line);
LLVM_ATTRIBUTE_WEAK void AnnotateIgnoreWritesEnd(const char *file, int line);
}
#endif

namespace llvm {
namespace sys {
  // True if Valgrind is controlling this process.
  bool RunningOnValgrind();

  // Discard valgrind's translation of code in the range [Addr .. Addr + Len).
  // Otherwise valgrind may continue to execute the old version of the code.
  void ValgrindDiscardTranslations(const void *Addr, size_t Len);

#if LLVM_ENABLE_THREADS != 0 && !defined(NDEBUG)
  // Thread Sanitizer is a valgrind tool that finds races in code.
  // See http://code.google.com/p/data-race-test/wiki/DynamicAnnotations .

  // This marker is used to define a happens-before arc. The race detector will
  // infer an arc from the begin to the end when they share the same pointer
  // argument.
  #define TsanHappensBefore(cv) \
    AnnotateHappensBefore(__FILE__, __LINE__, cv)

  // This marker defines the destination of a happens-before arc.
  #define TsanHappensAfter(cv) \
    AnnotateHappensAfter(__FILE__, __LINE__, cv)

  // Ignore any races on writes between here and the next TsanIgnoreWritesEnd.
  #define TsanIgnoreWritesBegin() \
    AnnotateIgnoreWritesBegin(__FILE__, __LINE__)

  // Resume checking for racy writes.
  #define TsanIgnoreWritesEnd() \
    AnnotateIgnoreWritesEnd(__FILE__, __LINE__)
#else
  #define TsanHappensBefore(cv)
  #define TsanHappensAfter(cv)
  #define TsanIgnoreWritesBegin()
  #define TsanIgnoreWritesEnd()
#endif
}
}

#endif
