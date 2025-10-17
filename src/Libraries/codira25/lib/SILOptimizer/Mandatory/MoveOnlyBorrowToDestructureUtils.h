/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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

//===--- MoveOnlyBorrowToDestructureUtils.h -------------------------------===//
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

#ifndef LANGUAGE_SILOPTIMIZER_MANDATORY_MOVEONLYBORROWTODESTRUCTURE_H
#define LANGUAGE_SILOPTIMIZER_MANDATORY_MOVEONLYBORROWTODESTRUCTURE_H

#include "language/Basic/FrozenMultiMap.h"
#include "language/SIL/FieldSensitivePrunedLiveness.h"
#include "language/SIL/StackList.h"
#include "language/SILOptimizer/Analysis/PostOrderAnalysis.h"

#include "toolchain/ADT/IntervalMap.h"

namespace language {
namespace siloptimizer {

class DiagnosticEmitter;

namespace borrowtodestructure {

class IntervalMapAllocator {
public:
  using Map = toolchain::IntervalMap<
      unsigned, SILValue,
      toolchain::IntervalMapImpl::NodeSizer<unsigned, SILValue>::LeafSize,
      toolchain::IntervalMapHalfOpenInfo<unsigned>>;

  using Allocator = Map::Allocator;

private:
  /// Lazily initialized allocator.
  std::optional<Allocator> allocator;

public:
  Allocator &get() {
    if (!allocator)
      allocator.emplace();
    return *allocator;
  }
};

struct Implementation;

} // namespace borrowtodestructure

class BorrowToDestructureTransform {
  friend borrowtodestructure::Implementation;

  using IntervalMapAllocator = borrowtodestructure::IntervalMapAllocator;

  IntervalMapAllocator &allocator;
  MarkUnresolvedNonCopyableValueInst *mmci;
  SILValue rootValue;
  DiagnosticEmitter &diagnosticEmitter;
  PostOrderAnalysis *poa;
  PostOrderFunctionInfo *pofi = nullptr;
  SmallVector<SILInstruction *, 8> createdDestructures;
  SmallVector<SILPhiArgument *, 8> createdPhiArguments;

public:
  BorrowToDestructureTransform(IntervalMapAllocator &allocator,
                               MarkUnresolvedNonCopyableValueInst *mmci,
                               SILValue rootValue,
                               DiagnosticEmitter &diagnosticEmitter,
                               PostOrderAnalysis *poa)
      : allocator(allocator), mmci(mmci), rootValue(rootValue),
        diagnosticEmitter(diagnosticEmitter), poa(poa) {}

  bool transform();

private:
  PostOrderFunctionInfo *getPostOrderFunctionInfo() {
    if (!pofi)
      pofi = poa->get(mmci->getFunction());
    return pofi;
  }
};

} // namespace siloptimizer
} // namespace language

#endif
