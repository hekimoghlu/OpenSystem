/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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

//===--- AccessSummaryDumper.cpp - Dump access summaries for functions -----===//
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

#define DEBUG_TYPE "sil-access-summary-dumper"
#include "language/SIL/SILArgument.h"
#include "language/SIL/SILFunction.h"
#include "language/SIL/SILInstruction.h"
#include "language/SIL/SILValue.h"
#include "language/SILOptimizer/Analysis/AccessSummaryAnalysis.h"
#include "language/SILOptimizer/PassManager/Passes.h"
#include "language/SILOptimizer/PassManager/Transforms.h"
#include "toolchain/Support/Debug.h"

using namespace language;

namespace {

/// Dumps summaries of kinds of accesses a function performs on its
/// @inout_aliasiable arguments.
class AccessSummaryDumper : public SILModuleTransform {

  void run() override {
    auto *analysis = PM->getAnalysis<AccessSummaryAnalysis>();

    for (auto &fn : *getModule()) {
      toolchain::outs() << "@" << fn.getName() << "\n";
      if (fn.empty()) {
        toolchain::outs() << "<unknown>\n";
        continue;
      }
      const AccessSummaryAnalysis::FunctionSummary &summary =
          analysis->getOrCreateSummary(&fn);
      summary.print(toolchain::outs(), &fn);
      toolchain::outs() << "\n";
    }
  }
};

} // end anonymous namespace

SILTransform *language::createAccessSummaryDumper() {
  return new AccessSummaryDumper();
}
