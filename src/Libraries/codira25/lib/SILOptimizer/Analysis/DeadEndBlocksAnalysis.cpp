/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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

//===--- DeadEndBlocksAnalysis.cpp ----------------------------------------===//
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

#include "language/SILOptimizer/Analysis/DeadEndBlocksAnalysis.h"
#include "language/SILOptimizer/OptimizerBridging.h"
#include "language/AST/Decl.h"
#include "language/Basic/Assertions.h"
#include "language/SIL/SILFunction.h"

using namespace language;

void DeadEndBlocksAnalysis::verify(DeadEndBlocks *deBlocks) const {
  // If the passed in deBlocks has not computed, there is nothing to check.
  if (!deBlocks->isComputed())
    return;

  // Then create our new dead end blocks instance so we can check the internal
  // state of our input against it.
  auto *fn = deBlocks->getFunction();
  DeadEndBlocks newBlocks(fn);

  // Make sure that all values that deBlocks thinks is unreachable are
  // actually unreachable.
  //
  // NOTE: We verify like this b/c DeadEndBlocks looks up state lazily so we
  // can only check the work we have done so far.
  for (auto &block : *fn) {
    if (deBlocks->isDeadEnd(&block)) {
      if (!newBlocks.isDeadEnd(&block)) {
        toolchain::errs() << "DeadEndBlocksAnalysis Error! Found dead end block "
                        "that is no longer a dead end block?!";
        toolchain_unreachable("standard error assertion");
      }
    } else {
      if (newBlocks.isDeadEnd(&block)) {
        toolchain::errs() << "DeadEndBlocksAnalysis Error! Found reachable block "
                        "that is no longer reachable?!";
        toolchain_unreachable("standard error assertion");
      }
    }
  }
}

//===----------------------------------------------------------------------===//
//                              Main Entry Point
//===----------------------------------------------------------------------===//

SILAnalysis *language::createDeadEndBlocksAnalysis(SILModule *) {
  return new DeadEndBlocksAnalysis();
}
