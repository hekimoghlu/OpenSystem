/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 24, 2022.
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

//===-- llvm/AutoUpgrade.h - AutoUpgrade Helpers ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  These functions are implemented by lib/VMCore/AutoUpgrade.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AUTOUPGRADE_H
#define LLVM_AUTOUPGRADE_H

namespace llvm {
  class Module;
  class GlobalVariable;
  class Function;
  class CallInst;

  /// This is a more granular function that simply checks an intrinsic function 
  /// for upgrading, and returns true if it requires upgrading. It may return
  /// null in NewFn if the all calls to the original intrinsic function
  /// should be transformed to non-function-call instructions.
  bool UpgradeIntrinsicFunction(Function *F, Function *&NewFn);

  /// This is the complement to the above, replacing a specific call to an 
  /// intrinsic function with a call to the specified new function.
  void UpgradeIntrinsicCall(CallInst *CI, Function *NewFn);
  
  /// This is an auto-upgrade hook for any old intrinsic function syntaxes 
  /// which need to have both the function updated as well as all calls updated 
  /// to the new function. This should only be run in a post-processing fashion 
  /// so that it can update all calls to the old function.
  void UpgradeCallsToIntrinsic(Function* F);

  /// This checks for global variables which should be upgraded. It returns true
  /// if it requires upgrading.
  bool UpgradeGlobalVariable(GlobalVariable *GV);
} // End llvm namespace

#endif
