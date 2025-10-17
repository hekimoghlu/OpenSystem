/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 4, 2022.
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

//===- BasicCalleeAnalysis.h - Determine callees per call site --*- C++ -*-===//
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

#ifndef LANGUAGE_SILOPTIMIZER_ANALYSIS_BASICCALLEEANALYSIS_H
#define LANGUAGE_SILOPTIMIZER_ANALYSIS_BASICCALLEEANALYSIS_H

#include "language/SILOptimizer/Analysis/Analysis.h"
#include "language/SIL/CalleeCache.h"
#include "language/SIL/SILFunction.h"
#include "language/SIL/SILInstruction.h"
#include "language/SIL/ApplySite.h"
#include "language/SIL/SILModule.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/PointerIntPair.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/TinyPtrVector.h"
#include "toolchain/Support/Allocator.h"

namespace language {

class BasicCalleeAnalysis : public SILAnalysis {
  SILModule &M;
  std::unique_ptr<CalleeCache> Cache;

public:

  BasicCalleeAnalysis(SILModule *M)
      : SILAnalysis(SILAnalysisKind::BasicCallee), M(*M), Cache(nullptr) {}

  ~BasicCalleeAnalysis() {}

  static bool classof(const SILAnalysis *S) {
    return S->getKind() == SILAnalysisKind::BasicCallee;
  }

  /// Invalidate all information in this analysis.
  virtual void invalidate() override {
    Cache.reset();
  }

  /// Invalidate all of the information for a specific function.
  virtual void invalidate(SILFunction *F, InvalidationKind K) override {
    // No invalidation needed because the analysis does not cache anything
    // per call-site in functions.
  }

  /// Notify the analysis about a newly created function.
  virtual void notifyAddedOrModifiedFunction(SILFunction *F) override {
    // Nothing to be done because the analysis does not cache anything
    // per call-site in functions.
  }

  /// Notify the analysis about a function which will be deleted from the
  /// module.
  virtual void notifyWillDeleteFunction(SILFunction *F) override {
    // No invalidation needed because the analysis does not cache anything per
    // call-site in functions.
  }

  /// Notify the analysis about changed witness or vtables.
  virtual void invalidateFunctionTables() override {
    Cache.reset();
  }

  LANGUAGE_DEBUG_DUMP;

  void print(toolchain::raw_ostream &os) const;

  void updateCache() {
    if (!Cache)
      Cache = std::make_unique<CalleeCache>(M);
  }

  CalleeList getCalleeList(FullApplySite FAS) {
    updateCache();
    return Cache->getCalleeList(FAS);
  }

  CalleeList getCalleeListOfValue(SILValue callee) {
    updateCache();
    return Cache->getCalleeListOfValue(callee);
  }

  CalleeList getDestructors(SILType type, bool isExactType) {
    updateCache();
    return Cache->getDestructors(type, isExactType);
  }

  MemoryBehavior getMemoryBehavior(FullApplySite as, bool observeRetains);

  CalleeCache *getCalleeCache() { return Cache.get(); }
};

bool isDeinitBarrier(SILInstruction *const instruction,
                     BasicCalleeAnalysis *bca);

} // end namespace language

#endif
