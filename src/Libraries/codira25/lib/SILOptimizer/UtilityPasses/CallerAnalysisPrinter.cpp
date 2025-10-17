/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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

//===--- CallerAnalysisPrinter.cpp - Caller Analysis test pass ------------===//
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
// This pass prints all the callsites of every function in the module.
//
//===----------------------------------------------------------------------===//

#include "language/SIL/SILFunction.h"
#include "language/SIL/SILModule.h"
#include "language/SILOptimizer/Analysis/CallerAnalysis.h"
#include "language/SILOptimizer/PassManager/Transforms.h"
#include "toolchain/Support/YAMLTraits.h"
#include "toolchain/Support/raw_ostream.h"

using namespace language;

#define DEBUG_TYPE "caller-analysis-printer"

namespace {

/// A pass that dumps the caller analysis state in yaml form. Intended to allow
/// for visualizing of the caller analysis via external data visualization and
/// analysis programs.
class CallerAnalysisPrinterPass : public SILModuleTransform {
  /// The entry point to the transformation.
  void run() override {
    auto *CA = getAnalysis<CallerAnalysis>();
    CA->print(toolchain::outs());
  }
};

} // end anonymous namespace

SILTransform *language::createCallerAnalysisPrinter() {
  return new CallerAnalysisPrinterPass();
}
