/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 30, 2022.
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

//===--- LoopAnalysis.h - SIL Loop Analysis ---------------------*- C++ -*-===//
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

#ifndef LANGUAGE_SILOPTIMIZER_ANALYSIS_LOOPINFOANALYSIS_H
#define LANGUAGE_SILOPTIMIZER_ANALYSIS_LOOPINFOANALYSIS_H

#include "language/SIL/CFG.h"
#include "language/SIL/LoopInfo.h"
#include "language/SIL/SILBasicBlock.h"
#include "language/SILOptimizer/Analysis/Analysis.h"

namespace language {
  class DominanceInfo;
  class SILLoop;
  class DominanceAnalysis;
}

namespace language {

/// Computes natural loop information for SIL basic blocks.
class SILLoopAnalysis : public FunctionAnalysisBase<SILLoopInfo> {
  DominanceAnalysis *DA;
public:
  SILLoopAnalysis(SILModule *)
      : FunctionAnalysisBase(SILAnalysisKind::Loop), DA(nullptr) {}

  static bool classof(const SILAnalysis *S) {
    return S->getKind() == SILAnalysisKind::Loop;
  }

  virtual bool shouldInvalidate(SILAnalysis::InvalidationKind K) override {
    return K & InvalidationKind::Branches;
  }

  // Computes loop information for the given function using dominance
  // information.
  virtual std::unique_ptr<SILLoopInfo>
  newFunctionAnalysis(SILFunction *F) override;

  virtual void initialize(SILPassManager *PM) override;
};

} // end namespace language

#endif
