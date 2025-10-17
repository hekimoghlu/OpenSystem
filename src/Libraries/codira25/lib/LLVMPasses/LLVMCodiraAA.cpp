/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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

//===--- LLVMCodiraAA.cpp - LLVM Alias Analysis for Codira ------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//
//===----------------------------------------------------------------------===//

#include "language/LLVMPasses/Passes.h"
#include "LLVMARCOpts.h"
#include "toolchain/Analysis/TargetLibraryInfo.h"
#include "toolchain/IR/LegacyPassManager.h" 
#include "toolchain/IR/Module.h"

using namespace toolchain;
using namespace language;

//===----------------------------------------------------------------------===//
//                           Alias Analysis Result
//===----------------------------------------------------------------------===//

static ModRefInfo getConservativeModRefForKind(const toolchain::Instruction &I) {
  switch (classifyInstruction(I)) {
#define KIND(Name, MemBehavior) case RT_ ## Name: return ModRefInfo:: MemBehavior;
#include "LLVMCodira.def"
  }

  toolchain_unreachable("Not a valid Instruction.");
}

ModRefInfo CodiraAAResult::getModRefInfo(const toolchain::CallBase *Call,
                                        const toolchain::MemoryLocation &Loc,
                                        toolchain::AAQueryInfo &AAQI) {
  // We know at compile time that certain entry points do not modify any
  // compiler-visible state ever. Quickly check if we have one of those
  // instructions and return if so.
  if (ModRefInfo::NoModRef == getConservativeModRefForKind(*Call))
    return ModRefInfo::NoModRef;

  // Otherwise, delegate to the rest of the AA ModRefInfo machinery.
  return AAResultBase::getModRefInfo(Call, Loc, AAQI);
}

//===----------------------------------------------------------------------===//
//                        Alias Analysis Wrapper Pass
//===----------------------------------------------------------------------===//

char CodiraAAWrapperPass::ID = 0;
INITIALIZE_PASS_BEGIN(CodiraAAWrapperPass, "language-aa",
                      "Codira Alias Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(CodiraAAWrapperPass, "language-aa",
                    "Codira Alias Analysis", false, true)

CodiraAAWrapperPass::CodiraAAWrapperPass() : ImmutablePass(ID) {
  initializeCodiraAAWrapperPassPass(*PassRegistry::getPassRegistry());
}

bool CodiraAAWrapperPass::doInitialization(Module &M) {
  Result.reset(new CodiraAAResult());
  return false;
}

bool CodiraAAWrapperPass::doFinalization(Module &M) {
  Result.reset();
  return false;
}

void CodiraAAWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
}

AnalysisKey CodiraAA::Key;

CodiraAAResult CodiraAA::run(toolchain::Function &F,
                           toolchain::FunctionAnalysisManager &AM) {
  return CodiraAAResult();
}

//===----------------------------------------------------------------------===//
//                           Top Level Entry Point
//===----------------------------------------------------------------------===//

toolchain::ImmutablePass *language::createCodiraAAWrapperPass() {
  return new CodiraAAWrapperPass();
}
