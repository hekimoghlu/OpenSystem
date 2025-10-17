/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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

//===- IntegerDivision.cpp - Unit tests for the integer division code -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/BasicBlock.h"
#include "llvm/GlobalValue.h"
#include "llvm/Function.h"
#include "llvm/IRBuilder.h"
#include "llvm/Module.h"
#include "llvm/Transforms/Utils/IntegerDivision.h"

using namespace llvm;

namespace {

TEST(IntegerDivision, SDiv) {
  LLVMContext &C(getGlobalContext());
  Module M("test division", C);
  IRBuilder<> Builder(C);

  SmallVector<Type*, 2> ArgTys(2, Builder.getInt32Ty());
  Function *F = Function::Create(FunctionType::get(Builder.getInt32Ty(),
                                                   ArgTys, false),
                                 GlobalValue::ExternalLinkage, "F", &M);
  assert(F->getArgumentList().size() == 2);

  BasicBlock *BB = BasicBlock::Create(C, "", F);
  Builder.SetInsertPoint(BB);

  Function::arg_iterator AI = F->arg_begin();
  Value *A = AI++;
  Value *B = AI++;

  Value *Div = Builder.CreateSDiv(A, B);
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::SDiv);

  Value *Ret = Builder.CreateRet(Div);

  expandDivision(cast<BinaryOperator>(Div));
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::AShr);

  Instruction* Quotient = dyn_cast<Instruction>(cast<User>(Ret)->getOperand(0));
  EXPECT_TRUE(Quotient && Quotient->getOpcode() == Instruction::Sub);
}

TEST(IntegerDivision, UDiv) {
  LLVMContext &C(getGlobalContext());
  Module M("test division", C);
  IRBuilder<> Builder(C);

  SmallVector<Type*, 2> ArgTys(2, Builder.getInt32Ty());
  Function *F = Function::Create(FunctionType::get(Builder.getInt32Ty(),
                                                   ArgTys, false),
                                 GlobalValue::ExternalLinkage, "F", &M);
  assert(F->getArgumentList().size() == 2);

  BasicBlock *BB = BasicBlock::Create(C, "", F);
  Builder.SetInsertPoint(BB);

  Function::arg_iterator AI = F->arg_begin();
  Value *A = AI++;
  Value *B = AI++;

  Value *Div = Builder.CreateUDiv(A, B);
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::UDiv);

  Value *Ret = Builder.CreateRet(Div);

  expandDivision(cast<BinaryOperator>(Div));
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::ICmp);

  Instruction* Quotient = dyn_cast<Instruction>(cast<User>(Ret)->getOperand(0));
  EXPECT_TRUE(Quotient && Quotient->getOpcode() == Instruction::PHI);
}

TEST(IntegerDivision, SRem) {
  LLVMContext &C(getGlobalContext());
  Module M("test remainder", C);
  IRBuilder<> Builder(C);

  SmallVector<Type*, 2> ArgTys(2, Builder.getInt32Ty());
  Function *F = Function::Create(FunctionType::get(Builder.getInt32Ty(),
                                                   ArgTys, false),
                                 GlobalValue::ExternalLinkage, "F", &M);
  assert(F->getArgumentList().size() == 2);

  BasicBlock *BB = BasicBlock::Create(C, "", F);
  Builder.SetInsertPoint(BB);

  Function::arg_iterator AI = F->arg_begin();
  Value *A = AI++;
  Value *B = AI++;

  Value *Rem = Builder.CreateSRem(A, B);
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::SRem);

  Value *Ret = Builder.CreateRet(Rem);

  expandRemainder(cast<BinaryOperator>(Rem));
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::AShr);

  Instruction* Remainder = dyn_cast<Instruction>(cast<User>(Ret)->getOperand(0));
  EXPECT_TRUE(Remainder && Remainder->getOpcode() == Instruction::Sub);
}

TEST(IntegerDivision, URem) {
  LLVMContext &C(getGlobalContext());
  Module M("test remainder", C);
  IRBuilder<> Builder(C);

  SmallVector<Type*, 2> ArgTys(2, Builder.getInt32Ty());
  Function *F = Function::Create(FunctionType::get(Builder.getInt32Ty(),
                                                   ArgTys, false),
                                 GlobalValue::ExternalLinkage, "F", &M);
  assert(F->getArgumentList().size() == 2);

  BasicBlock *BB = BasicBlock::Create(C, "", F);
  Builder.SetInsertPoint(BB);

  Function::arg_iterator AI = F->arg_begin();
  Value *A = AI++;
  Value *B = AI++;

  Value *Rem = Builder.CreateURem(A, B);
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::URem);

  Value *Ret = Builder.CreateRet(Rem);

  expandRemainder(cast<BinaryOperator>(Rem));
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::ICmp);

  Instruction* Remainder = dyn_cast<Instruction>(cast<User>(Ret)->getOperand(0));
  EXPECT_TRUE(Remainder && Remainder->getOpcode() == Instruction::Sub);
}

}
