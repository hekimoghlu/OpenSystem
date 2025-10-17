/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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

//===- Local.cpp - Unit tests for Local -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/BasicBlock.h"
#include "llvm/IRBuilder.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Transforms/Utils/Local.h"

#include "gtest/gtest.h"

using namespace llvm;

TEST(Local, RecursivelyDeleteDeadPHINodes) {
  LLVMContext &C(getGlobalContext());

  IRBuilder<> builder(C);

  // Make blocks
  BasicBlock *bb0 = BasicBlock::Create(C);
  BasicBlock *bb1 = BasicBlock::Create(C);

  builder.SetInsertPoint(bb0);
  PHINode    *phi = builder.CreatePHI(Type::getInt32Ty(C), 2);
  BranchInst *br0 = builder.CreateCondBr(builder.getTrue(), bb0, bb1);

  builder.SetInsertPoint(bb1);
  BranchInst *br1 = builder.CreateBr(bb0);

  phi->addIncoming(phi, bb0);
  phi->addIncoming(phi, bb1);

  // The PHI will be removed
  EXPECT_TRUE(RecursivelyDeleteDeadPHINode(phi));

  // Make sure the blocks only contain the branches
  EXPECT_EQ(&bb0->front(), br0);
  EXPECT_EQ(&bb1->front(), br1);

  builder.SetInsertPoint(bb0);
  phi = builder.CreatePHI(Type::getInt32Ty(C), 0);

  EXPECT_TRUE(RecursivelyDeleteDeadPHINode(phi));

  builder.SetInsertPoint(bb0);
  phi = builder.CreatePHI(Type::getInt32Ty(C), 0);
  builder.CreateAdd(phi, phi);

  EXPECT_TRUE(RecursivelyDeleteDeadPHINode(phi));

  bb0->dropAllReferences();
  bb1->dropAllReferences();
  delete bb0;
  delete bb1;
}
