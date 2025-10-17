/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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

//===--- DiagnosticDeadFunctionElimination.cpp ----------------------------===//
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
/// Delete functions that early diagnostic specialization passes mark as being
/// able to be DCE-ed if there are no further uses. This prevents later
/// diagnostic passes from emitting diagnostics both on the original function
/// and the diagnostic function.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sil-diagnostic-dead-function-eliminator"

#include "language/AST/SemanticAttrs.h"
#include "language/SIL/SILBuilder.h"
#include "language/SILOptimizer/PassManager/Passes.h"
#include "language/SILOptimizer/PassManager/Transforms.h"
#include "toolchain/Support/Debug.h"

using namespace language;

//===----------------------------------------------------------------------===//
//                         MARK: Top Level Entrypoint
//===----------------------------------------------------------------------===//

namespace {

struct DiagnosticDeadFunctionEliminator : SILFunctionTransform {
  void run() override {
    auto *fn = getFunction();

    // If an earlier pass asked us to eliminate the function body if it's
    // unused, and the function is in fact unused, do that now.
    if (!fn->hasSemanticsAttr(semantics::DELETE_IF_UNUSED) ||
        fn->getRefCount() != 0 ||
        isPossiblyUsedExternally(fn->getLinkage(),
                                 fn->getModule().isWholeModule())) {
      return;
    }

    TOOLCHAIN_DEBUG(toolchain::dbgs()
               << "===> Stubbifying unused function " << fn->getName()
               << "'s body that was marked for deletion\n");
    // Remove all non-entry blocks.
    auto entryBB = fn->begin();
    auto nextBB = std::next(entryBB);

    while (nextBB != fn->end()) {
      auto thisBB = nextBB;
      ++nextBB;
      thisBB->eraseFromParent();
    }

    // Rewrite the entry block to only contain an unreachable.
    auto loc = entryBB->begin()->getLoc();
    entryBB->eraseAllInstructions(fn->getModule());
    {
      SILBuilder b(&*entryBB);
      b.createUnreachable(loc);
    }

    // If the function has shared linkage, reduce this version to private
    // linkage, because we don't want the deleted-body form to win in any
    // ODR shootouts.
    if (fn->getLinkage() == SILLinkage::Shared) {
      fn->setLinkage(SILLinkage::Private);
      fn->setSerializedKind(IsNotSerialized);
    }

    invalidateAnalysis(SILAnalysis::InvalidationKind::FunctionBody);
  }
};

} // namespace

SILTransform *language::createDiagnosticDeadFunctionElimination() {
  return new DiagnosticDeadFunctionEliminator();
}
