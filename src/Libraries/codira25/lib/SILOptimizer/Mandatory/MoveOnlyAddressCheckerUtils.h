/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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

//===--- MoveOnlyAddressCheckerUtils.h ------------------------------------===//
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

#ifndef LANGUAGE_SILOPTIMIZER_MANDATORY_MOVEONLYADDRESSCHECKERUTILS_H
#define LANGUAGE_SILOPTIMIZER_MANDATORY_MOVEONLYADDRESSCHECKERUTILS_H

#include "MoveOnlyBorrowToDestructureUtils.h"

namespace language {

class PostOrderAnalysis;
class DeadEndBlocksAnalysis;

namespace siloptimizer {

class DiagnosticEmitter;

/// Searches for candidate mark must checks.
///
/// NOTE: To see if we emitted a diagnostic, use \p
/// diagnosticEmitter.getDiagnosticCount().
void searchForCandidateAddressMarkUnresolvedNonCopyableValueInsts(
    SILFunction *fn, PostOrderAnalysis *poa,
    toolchain::SmallSetVector<MarkUnresolvedNonCopyableValueInst *, 32>
        &moveIntroducersToProcess,
    DiagnosticEmitter &diagnosticEmitter);

struct MoveOnlyAddressChecker {
  SILFunction *fn;
  DiagnosticEmitter &diagnosticEmitter;
  borrowtodestructure::IntervalMapAllocator &allocator;
  DominanceInfo *domTree;
  PostOrderAnalysis *poa;
  DeadEndBlocksAnalysis *deadEndBlocksAnalysis;

  /// \returns true if we changed the IR. To see if we emitted a diagnostic, use
  /// \p diagnosticEmitter.getDiagnosticCount().
  bool check(toolchain::SmallSetVector<MarkUnresolvedNonCopyableValueInst *, 32>
                 &moveIntroducersToProcess);
  bool completeLifetimes();
};

} // namespace siloptimizer

} // namespace language

#endif
