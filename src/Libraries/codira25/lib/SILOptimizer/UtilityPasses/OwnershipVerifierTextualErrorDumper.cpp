/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 25, 2024.
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

//===--- OwnershipVerifierStateDumper.cpp ---------------------------------===//
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
///
/// This is a simple utility pass that verifies the ownership of all SILValue in
/// a module with a special flag set that causes the verifier to emit textual
/// errors instead of asserting. This is done one function at a time so that we
/// can number the errors as we emit them so in FileCheck tests, we can be 100%
/// sure we exactly matched the number of errors.
///
//===----------------------------------------------------------------------===//

#include "language/Basic/Assertions.h"
#include "language/SIL/BasicBlockUtils.h"
#include "language/SIL/SILFunction.h"
#include "language/SIL/SILInstruction.h"
#include "language/SILOptimizer/Analysis/DeadEndBlocksAnalysis.h"
#include "language/SILOptimizer/PassManager/Passes.h"
#include "language/SILOptimizer/PassManager/Transforms.h"

using namespace language;

//===----------------------------------------------------------------------===//
//                            Top Level Entrypoint
//===----------------------------------------------------------------------===//

namespace {

class OwnershipVerifierTextualErrorDumper : public SILFunctionTransform {
  void run() override {
    SILFunction *f = getFunction();
    auto *deBlocksAnalysis = getAnalysis<DeadEndBlocksAnalysis>();
    f->verifyOwnership(f->getModule().getOptions().OSSAVerifyComplete
                           ? nullptr
                           : deBlocksAnalysis->get(f));
  }
};

} // end anonymous namespace

SILTransform *language::createOwnershipVerifierTextualErrorDumper() {
  return new OwnershipVerifierTextualErrorDumper();
}
