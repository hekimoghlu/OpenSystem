/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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

//===--- GuaranteedPhiVerifier.cpp ----------------------------------------===//
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

#define DEBUG_TYPE "sil-guaranteed-phi-verifier"

#include "GuaranteedPhiVerifierPrivate.h"

using namespace language;

bool GuaranteedPhiVerifier::verifyDependentPhiLifetime(SILPhiArgument *phi,
                                                       SILValue baseValue) {
  // If the (phi, baseValue) pair was visited before, return.
  auto it = dependentPhiToBaseValueMap.find(phi);
  if (it != dependentPhiToBaseValueMap.end() &&
      it->second.find(baseValue) != it->second.end()) {
    return false;
  }
  // Insert (phi, baseValue) into the map.
  dependentPhiToBaseValueMap[phi].insert(baseValue);

  // Now, verify whether phi's lifetime is within baseValue's lifetime
  SmallVector<Operand *, 4> baseValConsumingUses(lookThroughBorrowedFromUser(baseValue)->getConsumingUses());
  // If the baseValue has no consuming uses, there is nothing more to verify
  if (baseValConsumingUses.empty())
    return false;

  SmallVector<Operand *, 4> phiUses(lookThroughBorrowedFromUser(phi)->getUses());
  LinearLifetimeChecker checker(deadEndBlocks);
  // newErrorBuilder is consumed at the end of the checkValue function.
  // Copy initial state from errorBuilder everytime
  LinearLifetimeChecker::ErrorBuilder newErrorBuilder = errorBuilder;
  // Verify whether the guaranteed phi lies within the lifetime of the base
  // value.
  auto linearLifetimeResult = checker.checkValue(
      baseValue, baseValConsumingUses, phiUses, newErrorBuilder);
  return linearLifetimeResult.getFoundError();
}

void GuaranteedPhiVerifier::verifyReborrows(BeginBorrowInst *borrow) {
  auto visitReborrowPhiBaseValuePair = [&](SILPhiArgument *phi,
                                           SILValue baseValue) {
    verifyDependentPhiLifetime(phi, baseValue);
  };
  visitExtendedReborrowPhiBaseValuePairs(borrow, visitReborrowPhiBaseValuePair);
}

void GuaranteedPhiVerifier::verifyGuaranteedForwardingPhis(
    BorrowedValue borrow) {
  auto visitGuaranteedPhiBaseValuePair = [&](SILPhiArgument *phi,
                                             SILValue baseValue) {
    verifyDependentPhiLifetime(phi, baseValue);
  };
  visitExtendedGuaranteedForwardingPhiBaseValuePairs(
      borrow, visitGuaranteedPhiBaseValuePair);
}
