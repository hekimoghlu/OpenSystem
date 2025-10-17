/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 6, 2022.
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

//===--- IsSelfRecursiveAnalysis.h ----------------------------------------===//
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

#ifndef LANGUAGE_SILOPTIMIZER_ISSELFRECURSIVEANALYSIS_H
#define LANGUAGE_SILOPTIMIZER_ISSELFRECURSIVEANALYSIS_H

#include "language/SILOptimizer/Analysis/Analysis.h"

namespace language {
class SILFunction;

class IsSelfRecursive {
  const SILFunction *f;
  bool didComputeValue = false;
  bool isSelfRecursive = false;

  void compute();

public:
  IsSelfRecursive(const SILFunction *f) : f(f) {}

  ~IsSelfRecursive();

  bool isComputed() const { return didComputeValue; }

  bool get() {
    if (!didComputeValue) {
      compute();
      didComputeValue = true;
    }
    return isSelfRecursive;
  }

  SILFunction *getFunction() { return const_cast<SILFunction *>(f); }
};

class IsSelfRecursiveAnalysis final
    : public FunctionAnalysisBase<IsSelfRecursive> {
public:
  IsSelfRecursiveAnalysis()
      : FunctionAnalysisBase<IsSelfRecursive>(
            SILAnalysisKind::IsSelfRecursive) {}

  IsSelfRecursiveAnalysis(const IsSelfRecursiveAnalysis &) = delete;
  IsSelfRecursiveAnalysis &operator=(const IsSelfRecursiveAnalysis &) = delete;

  static SILAnalysisKind getAnalysisKind() {
    return SILAnalysisKind::IsSelfRecursive;
  }

  static bool classof(const SILAnalysis *s) {
    return s->getKind() == SILAnalysisKind::IsSelfRecursive;
  }

  std::unique_ptr<IsSelfRecursive> newFunctionAnalysis(SILFunction *f) override {
    return std::make_unique<IsSelfRecursive>(f);
  }

  bool shouldInvalidate(SILAnalysis::InvalidationKind k) override {
    return k & InvalidationKind::Calls;
  }
};

} // namespace language

#endif
