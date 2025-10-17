/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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

//===--- LoopAnalysis.cpp - SIL Loop Analysis -----------------------------===//
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

#include "language/Basic/Assertions.h"
#include "language/SIL/Dominance.h"
#include "language/SILOptimizer/Analysis/DominanceAnalysis.h"
#include "language/SILOptimizer/Analysis/LoopAnalysis.h"
#include "language/SILOptimizer/PassManager/PassManager.h"
#include "toolchain/Support/Debug.h"

using namespace language;

std::unique_ptr<SILLoopInfo>
SILLoopAnalysis::newFunctionAnalysis(SILFunction *F) {
  assert(DA != nullptr && "Expect a valid dominance analysis");
  DominanceInfo *DT = DA->get(F);
  assert(DT != nullptr && "Expect a valid dominance information");
  return std::make_unique<SILLoopInfo>(F, DT);
}

void SILLoopAnalysis::initialize(SILPassManager *PM) {
  DA = PM->getAnalysis<DominanceAnalysis>();
}

SILAnalysis *language::createLoopAnalysis(SILModule *M) {
  return new SILLoopAnalysis(M);
}
