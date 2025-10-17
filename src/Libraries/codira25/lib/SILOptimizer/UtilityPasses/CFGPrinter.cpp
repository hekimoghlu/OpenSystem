/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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

//===--- CFGPrinter.cpp - CFG printer pass --------------------------------===//
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
// This file defines external functions that can be called to explicitly
// instantiate the CFG printer.
//
//===----------------------------------------------------------------------===//

#include "language/SILOptimizer/PassManager/Passes.h"
#include "language/SILOptimizer/PassManager/Transforms.h"
#include "language/SIL/CFG.h"
#include "language/SIL/SILBasicBlock.h"
#include "language/SIL/SILInstruction.h"
#include "language/SIL/SILFunction.h"
#include "toolchain/Support/CommandLine.h"

using namespace language;

//===----------------------------------------------------------------------===//
//                                  Options
//===----------------------------------------------------------------------===//

toolchain::cl::opt<std::string> SILViewCFGOnlyFun(
    "sil-view-cfg-only-function", toolchain::cl::init(""),
    toolchain::cl::desc("Only produce a graphviz file for this function"));

toolchain::cl::opt<std::string> SILViewCFGOnlyFuns(
    "sil-view-cfg-only-functions", toolchain::cl::init(""),
    toolchain::cl::desc("Only produce a graphviz file for the sil for the functions "
                   "whose name contains this substring"));

//===----------------------------------------------------------------------===//
//                              Top Level Driver
//===----------------------------------------------------------------------===//

namespace {

class SILCFGPrinter : public SILFunctionTransform {
  /// The entry point to the transformation.
  void run() override {
    SILFunction *F = getFunction();

    // If we are not supposed to dump view this cfg, return.
    if (!SILViewCFGOnlyFun.empty() && F && F->getName() != SILViewCFGOnlyFun)
      return;
    if (!SILViewCFGOnlyFuns.empty() && F &&
        !F->getName().contains(SILViewCFGOnlyFuns))
      return;

    F->viewCFG();
  }
};

} // end anonymous namespace

SILTransform *language::createCFGPrinter() {
  return new SILCFGPrinter();
}
