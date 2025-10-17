/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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

//===--- OwnershipConversionElimination.cpp -------------------------------===//
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

#include "SemanticARC/SemanticARCOpts.h"
#include "SemanticARCOptVisitor.h"
#include "language/SIL/LinearLifetimeChecker.h"

using namespace language;
using namespace semanticarc;

//===----------------------------------------------------------------------===//
//                            Top Level Entrypoint
//===----------------------------------------------------------------------===//

bool SemanticARCOptVisitor::visitUncheckedOwnershipConversionInst(
    UncheckedOwnershipConversionInst *uoci) {
  // Return false if we are supposed to only be running guaranteed opts.
  if (ctx.onlyMandatoryOpts)
    return false;

  // Then check if we are running tests and shouldn't perform this optimization
  // since we are testing something else.
  if (!ctx.shouldPerform(ARCTransformKind::OwnershipConversionElimPeephole))
    return false;

  // Otherwise, shrink our state space so that we only consider conversions from
  // owned or guaranteed to unowned. These are always legal and it is sometimes
  // convenient to insert them to avoid forwarding issues when RAUWing
  // values. So we eliminate them here.
  if (uoci->getConversionOwnershipKind() != OwnershipKind::Unowned)
    return false;

  auto op = uoci->getOperand();
  auto opKind = op->getOwnershipKind();
  if (opKind != OwnershipKind::Owned && opKind != OwnershipKind::Guaranteed)
    return false;

  // Ok, we can perform our optimization. First go through all of the uses of
  // uoci and see if they can accept our operand without any changes and that
  // they do not consume values.
  SmallVector<Operand *, 8> newUses;
  for (auto *use : uoci->getUses()) {
    if (use->isLifetimeEnding() || !use->canAcceptKind(opKind))
      return false;
    newUses.push_back(use);
  }

  // Ok, now we need to perform our lifetime check.
  SmallVector<Operand *, 8> consumingUses(op->getConsumingUses());
  LinearLifetimeChecker checker(&ctx.getDeadEndBlocks());
  if (!checker.validateLifetime(op, consumingUses, newUses))
    return false;

  // Otherwise, we can perform our rauw.
  eraseAndRAUWSingleValueInstruction(uoci, op);
  return true;
}
