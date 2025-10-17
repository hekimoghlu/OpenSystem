/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 1, 2021.
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

//===--- EpilogueARCMatcherDumper.cpp - Find Epilogue Releases ------------===//
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
///
/// \file
/// This pass finds the epilogue releases matched to each argument of the
/// function.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sil-epilogue-arc-dumper"
#include "language/SILOptimizer/PassManager/Passes.h"
#include "language/SIL/SILArgument.h"
#include "language/SIL/SILFunction.h"
#include "language/SIL/SILValue.h"
#include "language/SILOptimizer/Analysis/Analysis.h"
#include "language/SILOptimizer/Analysis/AliasAnalysis.h"
#include "language/SILOptimizer/Analysis/EpilogueARCAnalysis.h"
#include "language/SILOptimizer/Analysis/RCIdentityAnalysis.h"
#include "language/SILOptimizer/PassManager/Transforms.h"

using namespace language;

//===----------------------------------------------------------------------===//
//                              Top Level Driver
//===----------------------------------------------------------------------===//

namespace {

/// Find and dump the epilogue release instructions for the arguments.
class SILEpilogueARCMatcherDumper : public SILModuleTransform {
  void run() override {
    for (auto &F: *getModule()) {
      // Function is not definition.
      if (!F.isDefinition())
        continue;

      // Find the epilogue releases of each owned argument. 
      for (auto Arg : F.getArguments()) {
        auto *EA = PM->getAnalysis<EpilogueARCAnalysis>()->get(&F);
        toolchain::outs() <<"START: " <<  F.getName() << "\n";
        toolchain::outs() << *Arg;

        // Find the retain instructions for the argument.
        toolchain::SmallSetVector<SILInstruction *, 1> RelInsts = 
          EA->computeEpilogueARCInstructions(EpilogueARCContext::EpilogueARCKind::Release,
                                             Arg);
        for (auto I : RelInsts) {
          toolchain::outs() << *I << "\n";
        }

        // Find the release instructions for the argument.
        toolchain::SmallSetVector<SILInstruction *, 1> RetInsts = 
          EA->computeEpilogueARCInstructions(EpilogueARCContext::EpilogueARCKind::Retain,
                                             Arg);
        for (auto I : RetInsts) {
          toolchain::outs() << *I << "\n";
        }

        toolchain::outs() <<"FINISH: " <<  F.getName() << "\n";
      }
    }
  }

};
        
} // end anonymous namespace

SILTransform *language::createEpilogueARCMatcherDumper() {
  return new SILEpilogueARCMatcherDumper();
}
