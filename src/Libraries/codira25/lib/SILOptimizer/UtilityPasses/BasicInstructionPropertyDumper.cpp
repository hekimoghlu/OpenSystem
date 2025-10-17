/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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

//===--- BasicInstructionPropertyDumper.cpp -------------------------------===//
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

#include "language/SIL/SILFunction.h"
#include "language/SIL/SILInstruction.h"
#include "language/SIL/SILModule.h"
#include "language/SILOptimizer/PassManager/Transforms.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/raw_ostream.h"

using namespace language;

namespace {

class BasicInstructionPropertyDumper : public SILModuleTransform {
  void run() override {
    for (auto &Fn : *getModule()) {
      unsigned Count = 0;
      toolchain::outs() << "@" << Fn.getName() << "\n";
      for (auto &BB : Fn) {
        for (auto &I : BB) {
          toolchain::outs() << "Inst #: " << Count++ << "\n    " << I;
          toolchain::outs() << "    Mem Behavior: " << I.getMemoryBehavior() << "\n";
          toolchain::outs() << "    Release Behavior: " << I.getReleasingBehavior()
                       << "\n";
        }
      }
    }
  }

};

} // end anonymous namespace

SILTransform *language::createBasicInstructionPropertyDumper() {
  return new BasicInstructionPropertyDumper();
}
