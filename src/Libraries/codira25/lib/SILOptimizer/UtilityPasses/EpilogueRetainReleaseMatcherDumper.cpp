/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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

//===--- EpilogueRetainReleaseMatcherDumper.cpp - Find Epilogue Releases --===//
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

#define DEBUG_TYPE "sil-epilogue-release-dumper"
#include "language/SILOptimizer/PassManager/Passes.h"
#include "language/SIL/SILArgument.h"
#include "language/SIL/SILFunction.h"
#include "language/SIL/SILValue.h"
#include "language/SILOptimizer/Analysis/AliasAnalysis.h"
#include "language/SILOptimizer/Analysis/ARCAnalysis.h"
#include "language/SILOptimizer/Analysis/Analysis.h"
#include "language/SILOptimizer/Analysis/RCIdentityAnalysis.h"
#include "language/SILOptimizer/PassManager/Transforms.h"

using namespace language;

//===----------------------------------------------------------------------===//
//                              Top Level Driver
//===----------------------------------------------------------------------===//

namespace {

/// Find and dump the epilogue release instructions for the arguments.
class SILEpilogueRetainReleaseMatcherDumper : public SILModuleTransform {

  void run() override {
    auto *RCIA = getAnalysis<RCIdentityAnalysis>();
    for (auto &Fn: *getModule()) {
      // Function is not definition.
      if (!Fn.isDefinition())
        continue;

      auto *AA = PM->getAnalysis<AliasAnalysis>(&Fn);

      toolchain::outs() << "START: sil @" << Fn.getName() << "\n";

      // Handle @owned return value.
      ConsumedResultToEpilogueRetainMatcher RetMap(RCIA->get(&Fn), AA, &Fn); 
      for (auto &RI : RetMap)
        toolchain::outs() << *RI;

      // Handle @owned function arguments.
      ConsumedArgToEpilogueReleaseMatcher RelMap(RCIA->get(&Fn), &Fn); 
      // Iterate over arguments and dump their epilogue releases.
      for (auto Arg : Fn.getArguments()) {
        toolchain::outs() << *Arg;
        // Can not find an epilogue release instruction for the argument.
        for (auto &RI : RelMap.getReleasesForArgument(Arg))
          toolchain::outs() << *RI;
      }

      toolchain::outs() << "END: sil @" << Fn.getName() << "\n";
    }
  }

};
        
} // end anonymous namespace

SILTransform *language::createEpilogueRetainReleaseMatcherDumper() {
  return new SILEpilogueRetainReleaseMatcherDumper();
}
