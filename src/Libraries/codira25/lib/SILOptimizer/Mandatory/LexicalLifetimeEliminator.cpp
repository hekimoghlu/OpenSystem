/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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

//===--- LexicalLifetimeEliminator.cpp ------------------------------------===//
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

#define DEBUG_TYPE "sil-lexical-lifetime-eliminator"

#include "language/SILOptimizer/PassManager/Transforms.h"

using namespace language;

namespace {

class LexicalLifetimeEliminatorPass : public SILFunctionTransform {
  void run() override {
    auto *fn = getFunction();

    if (fn->forceEnableLexicalLifetimes())
      return;

    // If we are already canonical, we do not have any diagnostics to emit.
    if (fn->wasDeserializedCanonical())
      return;

    // If we have experimental late lexical lifetimes enabled, we do not want to
    // run this pass since we want lexical lifetimes to exist later in the
    // pipeline.
    if (fn->getModule().getOptions().LexicalLifetimes ==
        LexicalLifetimesOption::On)
      return;

    bool madeChange = false;
    for (auto &block : *fn) {
      for (auto &inst : block) {
        if (auto *bbi = dyn_cast<BeginBorrowInst>(&inst)) {
          if (bbi->isLexical()) {
            bbi->removeIsLexical();
            madeChange = true;
          }
          continue;
        }
        if (auto *mvi = dyn_cast<MoveValueInst>(&inst)) {
          if (mvi->isLexical()) {
            mvi->removeIsLexical();
            madeChange = true;
          }
          continue;
        }
        if (auto *asi = dyn_cast<AllocStackInst>(&inst)) {
          if (asi->isLexical()) {
            asi->removeIsLexical();
            madeChange = true;
          }
          continue;
        }
      }
    }

    if (madeChange) {
      invalidateAnalysis(SILAnalysis::InvalidationKind::Instructions);
    }
  }
};

} // anonymous namespace

SILTransform *language::createLexicalLifetimeEliminator() {
  return new LexicalLifetimeEliminatorPass();
}
