/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 12, 2022.
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

//===--- DiagnoseUnnecessaryPreconcurrencyImports.cpp ---------------------===//
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
/// This is run after TransferNonSendable and uses Sema infrastructure to
/// determine if in Sema or TransferNonSendable any of the preconcurrency import
/// statements were not used.
///
/// This only runs when RegionIsolation is enabled. If RegionIsolation is
/// disabled, we emit the unnecessary preconcurrency imports earlier during Sema
/// since no later diagnostics will be emitted.
///
/// NOTE: This needs to be a module pass and run after TransferNonSendable so we
/// can guarantee that we have run TransferNonSendable on all functions in our
/// module before this runs.
///
//===----------------------------------------------------------------------===//

#include "language/AST/SourceFile.h"
#include "language/Basic/Assertions.h"
#include "language/SILOptimizer/PassManager/Transforms.h"
#include "language/Sema/Concurrency.h"

using namespace language;

//===----------------------------------------------------------------------===//
//                         MARK: Top Level Entrypoint
//===----------------------------------------------------------------------===//

namespace {

class DiagnoseUnnecessaryPreconcurrencyImports : public SILModuleTransform {
  void run() override {
    // If region isolation is not enabled... return early.
    if (!getModule()->getASTContext().LangOpts.hasFeature(
            Feature::RegionBasedIsolation))
      return;

    std::vector<SourceFile *> data;
    for (auto &fn : *getModule()) {
      auto *sf = fn.getSourceFile();
      if (!sf) {
        continue;
      }

      data.push_back(sf);
    }

    // Sort unique by filename so our diagnostics are deterministic.
    //
    // TODO: If we cannot rely upon this, just sort by pointer address. Non
    // determinism emission of diagnostics isn't great but it isn't fatal.
    sortUnique(data, [](SourceFile *lhs, SourceFile *rhs) -> bool {
      return lhs->getBufferID() < rhs->getBufferID();
    });

    // At this point, we know that we have our sorted unique list of source
    // files.
    for (auto *sf : data) {
      diagnoseUnnecessaryPreconcurrencyImports(*sf);
    }
  }
};

} // namespace

SILTransform *language::createDiagnoseUnnecessaryPreconcurrencyImports() {
  return new DiagnoseUnnecessaryPreconcurrencyImports();
}
