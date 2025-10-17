/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 4, 2022.
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

//===--- FunctionOrder.h - Utilities for function ordering  -----*- C++ -*-===//
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

#ifndef LANGUAGE_SILOPTIMIZER_ANALYSIS_FUNCTIONORDER_H
#define LANGUAGE_SILOPTIMIZER_ANALYSIS_FUNCTIONORDER_H

#include "language/SILOptimizer/Analysis/BasicCalleeAnalysis.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/SetVector.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/TinyPtrVector.h"

namespace language {

class BasicCalleeAnalysis;
class SILFunction;
class SILModule;

class BottomUpFunctionOrder {
public:
  typedef TinyPtrVector<SILFunction *> SCC;

private:
  SILModule &M;
  toolchain::SmallVector<SCC, 32> TheSCCs;
  toolchain::SmallVector<SILFunction *, 32> TheFunctions;

  // The callee analysis we use to determine the callees at each call site.
  BasicCalleeAnalysis *BCA;

  unsigned NextDFSNum;
  toolchain::DenseMap<SILFunction *, unsigned> DFSNum;
  toolchain::DenseMap<SILFunction *, unsigned> MinDFSNum;
  toolchain::SmallSetVector<SILFunction *, 4> DFSStack;

public:
  BottomUpFunctionOrder(SILModule &M, BasicCalleeAnalysis *BCA)
      : M(M), BCA(BCA), NextDFSNum(0) {}

  /// Get the SCCs in bottom-up order.
  ArrayRef<SCC> getSCCs() {
    if (!TheSCCs.empty())
      return TheSCCs;

    FindSCCs(M);
    return TheSCCs;
  }

  /// Get a flattened view of all functions in all the SCCs in
  /// bottom-up order
  ArrayRef<SILFunction *> getFunctions() {
    if (!TheFunctions.empty())
      return TheFunctions;

    for (auto SCC : getSCCs())
      for (auto *F : SCC)
        TheFunctions.push_back(F);

    return TheFunctions;
  }

private:
  void DFS(SILFunction *F);
  void FindSCCs(SILModule &M);
};

} // end namespace language

#endif
