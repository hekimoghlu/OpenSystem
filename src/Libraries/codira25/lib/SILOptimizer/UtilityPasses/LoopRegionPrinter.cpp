/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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

//===--- LoopRegionPrinter.cpp --------------------------------------------===//
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
//
// Simple pass for testing the new loop region dumper analysis. Prints out
// information suitable for checking with filecheck.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sil-loop-region-printer"

#include "language/SILOptimizer/Analysis/LoopRegionAnalysis.h"
#include "language/SILOptimizer/PassManager/Passes.h"
#include "language/SILOptimizer/PassManager/Transforms.h"
#include "toolchain/Support/CommandLine.h"

using namespace language;

static toolchain::cl::opt<std::string>
    SILViewCFGOnlyFun("sil-loop-region-view-cfg-only-function",
                      toolchain::cl::init(""),
                      toolchain::cl::desc("Only produce a graphviz file for the "
                                     "loop region info of this function"));

static toolchain::cl::opt<std::string>
    SILViewCFGOnlyFuns("sil-loop-region-view-cfg-only-functions",
                       toolchain::cl::init(""),
                       toolchain::cl::desc("Only produce a graphviz file for the "
                                      "loop region info for the functions "
                                      "whose name contains this substring"));

namespace {

class LoopRegionViewText : public SILModuleTransform {
  void run() override {
    invalidateAll();
    auto *lra = PM->getAnalysis<LoopRegionAnalysis>();

    for (auto &fn : *getModule()) {
      if (fn.isExternalDeclaration())
        continue;
      if (!SILViewCFGOnlyFun.empty() && fn.getName() != SILViewCFGOnlyFun)
        continue;
      if (!SILViewCFGOnlyFuns.empty() &&
          !fn.getName().contains(SILViewCFGOnlyFuns))
        continue;

      // Ok, we are going to analyze this function. Invalidate all state
      // associated with it so we recompute the loop regions.
      toolchain::outs() << "Start @" << fn.getName() << "@\n";
      lra->get(&fn)->dump();
      toolchain::outs() << "End @" << fn.getName() << "@\n";
      toolchain::outs().flush();
    }
  }
};

class LoopRegionViewCFG : public SILModuleTransform {
  void run() override {
    invalidateAll();
    auto *lra = PM->getAnalysis<LoopRegionAnalysis>();

    for (auto &fn : *getModule()) {
      if (fn.isExternalDeclaration())
        continue;
      if (!SILViewCFGOnlyFun.empty() && fn.getName() != SILViewCFGOnlyFun)
        continue;
      if (!SILViewCFGOnlyFuns.empty() &&
          !fn.getName().contains(SILViewCFGOnlyFuns))
        continue;

      // Ok, we are going to analyze this function. Invalidate all state
      // associated with it so we recompute the loop regions.
      lra->get(&fn)->viewLoopRegions();
    }
  }
};

} // end anonymous namespace

SILTransform *language::createLoopRegionViewText() {
  return new LoopRegionViewText();
}

SILTransform *language::createLoopRegionViewCFG() {
  return new LoopRegionViewCFG();
}
