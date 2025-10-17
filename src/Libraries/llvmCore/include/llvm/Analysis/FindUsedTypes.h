/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 18, 2023.
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

//===- llvm/Analysis/FindUsedTypes.h - Find all Types in use ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is used to seek out all of the types in use by the program.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_FINDUSEDTYPES_H
#define LLVM_ANALYSIS_FINDUSEDTYPES_H

#include "llvm/ADT/SetVector.h"
#include "llvm/Pass.h"

namespace llvm {

class Type;
class Value;

class FindUsedTypes : public ModulePass {
  SetVector<Type *> UsedTypes;
public:
  static char ID; // Pass identification, replacement for typeid
  FindUsedTypes() : ModulePass(ID) {
    initializeFindUsedTypesPass(*PassRegistry::getPassRegistry());
  }

  /// getTypes - After the pass has been run, return the set containing all of
  /// the types used in the module.
  ///
  const SetVector<Type *> &getTypes() const { return UsedTypes; }

  /// Print the types found in the module.  If the optional Module parameter is
  /// passed in, then the types are printed symbolically if possible, using the
  /// symbol table from the module.
  ///
  void print(raw_ostream &o, const Module *M) const;

private:
  /// IncorporateType - Incorporate one type and all of its subtypes into the
  /// collection of used types.
  ///
  void IncorporateType(Type *Ty);

  /// IncorporateValue - Incorporate all of the types used by this value.
  ///
  void IncorporateValue(const Value *V);

public:
  /// run - This incorporates all types used by the specified module
  bool runOnModule(Module &M);

  /// getAnalysisUsage - We do not modify anything.
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
};

} // End llvm namespace

#endif
