/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 21, 2024.
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

//===- llvm/unittest/Bitcode/BitReaderTest.cpp - Tests for BitReader ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

static Module *makeLLVMModule() {
  Module* Mod = new Module("test-mem", getGlobalContext());

  FunctionType* FuncTy =
    FunctionType::get(Type::getVoidTy(Mod->getContext()), false);
  Function* Func = Function::Create(FuncTy,GlobalValue::ExternalLinkage,
                                    "func", Mod);

  BasicBlock* Entry = BasicBlock::Create(Mod->getContext(), "entry", Func);
  new UnreachableInst(Mod->getContext(), Entry);

  BasicBlock* BB = BasicBlock::Create(Mod->getContext(), "bb", Func);
  new UnreachableInst(Mod->getContext(), BB);

  PointerType* Int8Ptr = Type::getInt8PtrTy(Mod->getContext());
  new GlobalVariable(*Mod, Int8Ptr, /*isConstant=*/true,
                     GlobalValue::ExternalLinkage,
                     BlockAddress::get(BB), "table");

  return Mod;
}

static void writeModuleToBuffer(SmallVectorImpl<char> &Buffer) {
  Module *Mod = makeLLVMModule();
  raw_svector_ostream OS(Buffer);
  WriteBitcodeToFile(Mod, OS);
}

TEST(BitReaderTest, MaterializeFunctionsForBlockAddr) { // PR11677
  SmallString<1024> Mem;
  writeModuleToBuffer(Mem);
  MemoryBuffer *Buffer = MemoryBuffer::getMemBuffer(Mem.str(), "test", false);
  std::string errMsg;
  Module *m = getLazyBitcodeModule(Buffer, getGlobalContext(), &errMsg);
  PassManager passes;
  passes.add(createVerifierPass());
  passes.run(*m);
}

}
}
