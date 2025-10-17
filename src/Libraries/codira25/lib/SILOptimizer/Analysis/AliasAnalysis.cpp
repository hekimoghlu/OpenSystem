/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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

//===--- AliasAnalysis.cpp - SIL Alias Analysis ---------------------------===//
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

#define DEBUG_TYPE "sil-aa"
#include "language/SILOptimizer/Analysis/AliasAnalysis.h"
#include "language/Basic/Assertions.h"
#include "language/SIL/SILFunction.h"
#include "language/SILOptimizer/PassManager/PassManager.h"
#include "toolchain/Support/Debug.h"
#include "language/SILOptimizer/OptimizerBridging.h"

using namespace language;

// Bridging functions.
static BridgedAliasAnalysis::InitFn initFunction = nullptr;
static BridgedAliasAnalysis::DestroyFn destroyFunction = nullptr;
static BridgedAliasAnalysis::GetMemEffectFn getMemEffectsFunction = nullptr;
static BridgedAliasAnalysis::Escaping2InstFn isObjReleasedFunction = nullptr;
static BridgedAliasAnalysis::Escaping2ValIntFn isAddrVisibleFromObjFunction = nullptr;
static BridgedAliasAnalysis::MayAliasFn mayAliasFunction = nullptr;

void AliasAnalysis::initCodiraSpecificData() {
  if (initFunction)
    initFunction({this}, sizeof(languageSpecificData));
}

AliasAnalysis::~AliasAnalysis() {
  if (destroyFunction)
    destroyFunction({this});
}

bool AliasAnalysis::canApplyDecrementRefCount(FullApplySite FAS, SILValue Ptr) {
  // Treat applications of no-return functions as decrementing ref counts. This
  // causes the apply to become a sink barrier for ref count increments.
  if (FAS.isCalleeNoReturn())
    return true;

  /// If the pointer cannot escape to the function we are done.
  bool result = isObjectReleasedByInst(Ptr, FAS.getInstruction());
  return result;
}

bool AliasAnalysis::canBuiltinDecrementRefCount(BuiltinInst *BI, SILValue Ptr) {
  return isObjectReleasedByInst(Ptr, BI);
}

namespace {

class AliasAnalysisContainer : public FunctionAnalysisBase<AliasAnalysis> {
  SILPassManager *PM = nullptr;

public:
  AliasAnalysisContainer() : FunctionAnalysisBase(SILAnalysisKind::Alias) {}

  virtual bool shouldInvalidate(SILAnalysis::InvalidationKind K) override {
    return K & InvalidationKind::Instructions;
  }

  virtual void invalidate(SILFunction *f,
                          SILAnalysis::InvalidationKind k) override {
    if (k & InvalidationKind::Effects) {
      FunctionAnalysisBase::invalidate();
    } else {
      FunctionAnalysisBase::invalidate(f, k);
    }
  }

  // Computes loop information for the given function using dominance
  // information.
  virtual std::unique_ptr<AliasAnalysis>
  newFunctionAnalysis(SILFunction *F) override {
    assert(PM && "dependent analysis not initialized");
    return std::make_unique<AliasAnalysis>(PM);
  }

  virtual void initialize(SILPassManager *PM) override {
    this->PM = PM;
  }
};

} // end anonymous namespace

SILAnalysis *language::createAliasAnalysis(SILModule *M) {
  return new AliasAnalysisContainer();
}

//===----------------------------------------------------------------------===//
//                            Codira Bridging
//===----------------------------------------------------------------------===//

void BridgedAliasAnalysis::registerAnalysis(InitFn initFn,
                                            DestroyFn destroyFn,
                                            GetMemEffectFn getMemEffectsFn,
                                            Escaping2InstFn isObjReleasedFn,
                                            Escaping2ValIntFn isAddrVisibleFromObjFn,
                                            MayAliasFn mayAliasFn) {
  initFunction = initFn;
  destroyFunction = destroyFn;
  getMemEffectsFunction = getMemEffectsFn;
  isObjReleasedFunction = isObjReleasedFn;
  isAddrVisibleFromObjFunction = isAddrVisibleFromObjFn;
  mayAliasFunction = mayAliasFn;
}

MemoryBehavior AliasAnalysis::computeMemoryBehavior(SILInstruction *toInst, SILValue addr) {
  if (getMemEffectsFunction) {
    return (MemoryBehavior)getMemEffectsFunction({PM->getCodiraPassInvocation()},
                                                 {this},
                                                 {addr},
                                                 {toInst->asSILNode()});
  }
  return MemoryBehavior::MayHaveSideEffects;
}

bool AliasAnalysis::isObjectReleasedByInst(SILValue obj, SILInstruction *inst) {
  if (isObjReleasedFunction) {
    return isObjReleasedFunction({PM->getCodiraPassInvocation()}, {this}, {obj}, {inst->asSILNode()});
  }
  return true;
}

bool AliasAnalysis::isAddrVisibleFromObject(SILValue addr, SILValue obj) {
  if (isAddrVisibleFromObjFunction) {
    return isAddrVisibleFromObjFunction({PM->getCodiraPassInvocation()}, {this}, {addr}, {obj});
  }
  return true;
}

bool AliasAnalysis::mayAlias(SILValue lhs, SILValue rhs) {
  if (mayAliasFunction) {
    return mayAliasFunction({PM->getCodiraPassInvocation()}, {this}, {lhs}, {rhs});
  }
  return true;
}
