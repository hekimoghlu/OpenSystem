/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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

//===--- DominanceAnalysis.h - SIL Dominance Analysis -----------*- C++ -*-===//
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

#ifndef LANGUAGE_SILOPTIMIZER_ANALYSIS_DOMINANCEANALYSIS_H
#define LANGUAGE_SILOPTIMIZER_ANALYSIS_DOMINANCEANALYSIS_H

#include "language/SIL/SILInstruction.h"
#include "language/SIL/Dominance.h"
#include "language/SILOptimizer/Analysis/Analysis.h"

namespace language {
class SILModule;
class SILInstruction;

class DominanceAnalysis : public FunctionAnalysisBase<DominanceInfo> {
protected:
  virtual void verify(DominanceInfo *DI) const override {
    if (DI->roots().empty())
      return;
    DI->verify();
  }

public:
  DominanceAnalysis()
      : FunctionAnalysisBase<DominanceInfo>(SILAnalysisKind::Dominance) {}

  DominanceAnalysis(const DominanceAnalysis &) = delete;
  DominanceAnalysis &operator=(const DominanceAnalysis &) = delete;

  static SILAnalysisKind getAnalysisKind() {
    return SILAnalysisKind::Dominance;
  }

  static bool classof(const SILAnalysis *S) {
    return S->getKind() == SILAnalysisKind::Dominance;
  }

  std::unique_ptr<DominanceInfo> newFunctionAnalysis(SILFunction *F) override {
    return std::make_unique<DominanceInfo>(F);
  }

  virtual bool shouldInvalidate(SILAnalysis::InvalidationKind K) override {
    return K & InvalidationKind::Branches;
  }
};

class PostDominanceAnalysis : public FunctionAnalysisBase<PostDominanceInfo> {
protected:
  virtual void verify(PostDominanceInfo *PDI) const override {
    if (PDI->roots().empty())
      return;
    PDI->verify();
  }

public:
  PostDominanceAnalysis()
      : FunctionAnalysisBase<PostDominanceInfo>(
            SILAnalysisKind::PostDominance) {}

  PostDominanceAnalysis(const PostDominanceAnalysis &) = delete;
  PostDominanceAnalysis &operator=(const PostDominanceAnalysis &) = delete;

  static SILAnalysisKind getAnalysisKind() {
    return SILAnalysisKind::PostDominance;
  }

  static bool classof(const SILAnalysis *S) {
    return S->getKind() == SILAnalysisKind::PostDominance;
  }

  std::unique_ptr<PostDominanceInfo>
  newFunctionAnalysis(SILFunction *F) override {
    return std::make_unique<PostDominanceInfo>(F);
  }

  virtual bool shouldInvalidate(SILAnalysis::InvalidationKind K) override {
    return K & InvalidationKind::Branches;
  }
};

} // end namespace language



#endif
