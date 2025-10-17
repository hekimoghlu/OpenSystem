/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 22, 2023.
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

//===--- Analysis.cpp - Codira Analysis ------------------------------------===//
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

#define DEBUG_TYPE "sil-analysis"
#include "language/SILOptimizer/Analysis/Analysis.h"
#include "language/AST/Module.h"
#include "language/AST/SILOptions.h"
#include "language/Basic/Assertions.h"
#include "language/SIL/SILFunction.h"
#include "language/SIL/SILModule.h"
#include "language/SILOptimizer/Analysis/BasicCalleeAnalysis.h"
#include "language/SILOptimizer/Analysis/ClassHierarchyAnalysis.h"
#include "language/SILOptimizer/Analysis/DominanceAnalysis.h"
#include "language/SILOptimizer/Analysis/IVAnalysis.h"
#include "language/SILOptimizer/Analysis/NonLocalAccessBlockAnalysis.h"
#include "language/SILOptimizer/Analysis/PostOrderAnalysis.h"
#include "language/SILOptimizer/Analysis/ProtocolConformanceAnalysis.h"
#include "language/SILOptimizer/Utils/InstOptUtils.h"
#include "toolchain/ADT/Statistic.h"
#include "toolchain/Support/Debug.h"

using namespace language;

void SILAnalysis::verifyFunction(SILFunction *F) {
  // Only functions with bodies can be analyzed by the analysis.
  assert(F->isDefinition() && "Can't analyze external functions");
}

SILAnalysis *language::createDominanceAnalysis(SILModule *) {
  return new DominanceAnalysis();
}

SILAnalysis *language::createPostDominanceAnalysis(SILModule *) {
  return new PostDominanceAnalysis();
}

SILAnalysis *language::createInductionVariableAnalysis(SILModule *M) {
  return new IVAnalysis(M);
}

SILAnalysis *language::createPostOrderAnalysis(SILModule *M) {
  return new PostOrderAnalysis();
}

SILAnalysis *language::createClassHierarchyAnalysis(SILModule *M) {
  return new ClassHierarchyAnalysis(M);
}

SILAnalysis *language::createBasicCalleeAnalysis(SILModule *M) {
  return new BasicCalleeAnalysis(M);
}

SILAnalysis *language::createProtocolConformanceAnalysis(SILModule *M) {
  return new ProtocolConformanceAnalysis(M);
}

SILAnalysis *language::createNonLocalAccessBlockAnalysis(SILModule *M) {
  return new NonLocalAccessBlockAnalysis();
}
