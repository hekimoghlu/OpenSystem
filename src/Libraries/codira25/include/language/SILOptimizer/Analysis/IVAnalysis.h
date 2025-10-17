/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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

//===--- IVAnalysis.h - SIL IV Analysis -------------------------*- C++ -*-===//
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

#ifndef LANGUAGE_SILOPTIMIZER_ANALYSIS_IVANALYSIS_H
#define LANGUAGE_SILOPTIMIZER_ANALYSIS_IVANALYSIS_H

#include "language/SILOptimizer/Analysis/Analysis.h"
#include "language/SIL/SILArgument.h"
#include "language/SIL/SILValue.h"
#include "language/SILOptimizer/Utils/SCCVisitor.h"
#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/Debug.h"

namespace language {

class IVInfo : public SCCVisitor<IVInfo> {
public:
  typedef toolchain::SmallVectorImpl<SILNode *> SCCType;
  friend SCCVisitor;

public:

  /// A descriptor for an induction variable comprised of a header argument
  /// (phi node) and an increment by an integer literal.
  class IVDesc {
  public:
    BuiltinInst *Inc;
    IntegerLiteralInst *IncVal;

    IVDesc() : Inc(nullptr), IncVal(nullptr) {}
    IVDesc(BuiltinInst *AI, IntegerLiteralInst *I) : Inc(AI), IncVal(I) {}

    operator bool() { return Inc != nullptr && IncVal != nullptr; }
    static IVDesc invalidIV() { return IVDesc(); }
  };

  IVInfo(SILFunction &F) : SCCVisitor(F) {
    run();
  }

  bool isInductionVariable(ValueBase *IV) {
    auto End = InductionVariableMap.end();
    auto Found = InductionVariableMap.find(IV);
    return Found != End;
  }

  SILArgument *getInductionVariableHeader(ValueBase *IV) {
    assert(isInductionVariable(IV) && "Expected induction variable!");

    return InductionVariableMap.find(IV)->second;
  }

  IVDesc getInductionDesc(SILArgument *Arg) {
    toolchain::DenseMap<const ValueBase *, IVDesc>::iterator CI =
        InductionInfoMap.find(Arg);
    if (CI == InductionInfoMap.end())
      return IVDesc::invalidIV();
    return CI->second;
  }

private:
  // Map from an element of an induction sequence to the header.
  toolchain::DenseMap<const ValueBase *, SILArgument *> InductionVariableMap;

  // Map from an induction variable header to the induction descriptor.
  toolchain::DenseMap<const ValueBase *, IVDesc> InductionInfoMap;

  SILArgument *isInductionSequence(SCCType &SCC);
  void visit(SCCType &SCC);
};

class IVAnalysis final : public FunctionAnalysisBase<IVInfo> {
public:
  IVAnalysis(SILModule *)
      : FunctionAnalysisBase<IVInfo>(SILAnalysisKind::InductionVariable) {}
  IVAnalysis(const IVAnalysis &) = delete;
  IVAnalysis &operator=(const IVAnalysis &) = delete;

  static bool classof(const SILAnalysis *S) {
    return S->getKind() == SILAnalysisKind::InductionVariable;
  }

  std::unique_ptr<IVInfo> newFunctionAnalysis(SILFunction *F) override {
    return std::make_unique<IVInfo>(*F);
  }

  /// For now we always invalidate.
  virtual bool shouldInvalidate(SILAnalysis::InvalidationKind K) override {
    return true;
  }
};

}

#endif
