/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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

//===- ProfilingUtils.h - Helper functions shared by profilers --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a few helper functions which are used by profile
// instrumentation code to instrument the code.  This allows the profiler pass
// to worry about *what* to insert, and these functions take care of *how* to do
// it.
//
//===----------------------------------------------------------------------===//

#ifndef PROFILINGUTILS_H
#define PROFILINGUTILS_H

namespace llvm {
  class BasicBlock;
  class Function;
  class GlobalValue;
  class Module;
  class PointerType;

  void InsertProfilingInitCall(Function *MainFn, const char *FnName,
                               GlobalValue *Arr = 0,
                               PointerType *arrayType = 0);
  void IncrementCounterInBlock(BasicBlock *BB, unsigned CounterNum,
                               GlobalValue *CounterArray,
                               bool beginning = true);
  void InsertProfilingShutdownCall(Function *Callee, Module *Mod);
}

#endif
