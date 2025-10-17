/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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

//===-- llvm/Analysis/Lint.h - LLVM IR Lint ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines lint interfaces that can be used for some sanity checking
// of input to the system, and for checking that transformations
// haven't done something bad. In contrast to the Verifier, the Lint checker
// checks for undefined behavior or constructions with likely unintended
// behavior.
//
// To see what specifically is checked, look at Lint.cpp
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LINT_H
#define LLVM_ANALYSIS_LINT_H

namespace llvm {

class FunctionPass;
class Module;
class Function;

/// @brief Create a lint pass.
///
/// Check a module or function.
FunctionPass *createLintPass();

/// @brief Check a module.
///
/// This should only be used for debugging, because it plays games with
/// PassManagers and stuff.
void lintModule(
  const Module &M    ///< The module to be checked
);

// lintFunction - Check a function.
void lintFunction(
  const Function &F  ///< The function to be checked
);

} // End llvm namespace

#endif
