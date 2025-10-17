/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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

//===-- ModuleUtils.cpp - Functions to manipulate Modules -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions perform manipulations on Modules.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/IRBuilder.h"
#include "llvm/Module.h"

using namespace llvm;

static void appendToGlobalArray(const char *Array, 
                                Module &M, Function *F, int Priority) {
  IRBuilder<> IRB(M.getContext());
  FunctionType *FnTy = FunctionType::get(IRB.getVoidTy(), false);
  StructType *Ty = StructType::get(
      IRB.getInt32Ty(), PointerType::getUnqual(FnTy), NULL);

  Constant *RuntimeCtorInit = ConstantStruct::get(
      Ty, IRB.getInt32(Priority), F, NULL);

  // Get the current set of static global constructors and add the new ctor
  // to the list.
  SmallVector<Constant *, 16> CurrentCtors;
  if (GlobalVariable * GVCtor = M.getNamedGlobal(Array)) {
    if (Constant *Init = GVCtor->getInitializer()) {
      unsigned n = Init->getNumOperands();
      CurrentCtors.reserve(n + 1);
      for (unsigned i = 0; i != n; ++i)
        CurrentCtors.push_back(cast<Constant>(Init->getOperand(i)));
    }
    GVCtor->eraseFromParent();
  }

  CurrentCtors.push_back(RuntimeCtorInit);

  // Create a new initializer.
  ArrayType *AT = ArrayType::get(RuntimeCtorInit->getType(),
                                 CurrentCtors.size());
  Constant *NewInit = ConstantArray::get(AT, CurrentCtors);

  // Create the new global variable and replace all uses of
  // the old global variable with the new one.
  (void)new GlobalVariable(M, NewInit->getType(), false,
                           GlobalValue::AppendingLinkage, NewInit, Array);
}

void llvm::appendToGlobalCtors(Module &M, Function *F, int Priority) {
  appendToGlobalArray("llvm.global_ctors", M, F, Priority);
}

void llvm::appendToGlobalDtors(Module &M, Function *F, int Priority) {
  appendToGlobalArray("llvm.global_dtors", M, F, Priority);
}
