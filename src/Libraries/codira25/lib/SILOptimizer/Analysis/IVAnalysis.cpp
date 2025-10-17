/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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

//===--- IVAnalysis.cpp - SIL IV Analysis ---------------------------------===//
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

#include "language/SILOptimizer/Analysis/IVAnalysis.h"
#include "language/Basic/Assertions.h"
#include "language/SIL/PatternMatch.h"
#include "language/SIL/SILInstruction.h"
#include "language/SIL/SILValue.h"

using namespace language;
using namespace language::PatternMatch;

#if !defined(NDEBUG)
static bool inSCC(ValueBase *value, IVInfo::SCCType &SCC) {
  for (SILNode *node : SCC) {
    if (node == value)
      return true;
  }
  return false;
}
#endif

// For now, we'll consider only the simplest induction variables:
// - Exactly one element in the cycle must be a SILArgument.
// - Only a single increment by a literal.
//
// In other words many valid things that could be considered induction
// variables are disallowed at this point.
SILArgument *IVInfo::isInductionSequence(SCCType &SCC) {
  // Ignore SCCs of size 1 for now. Some of these are derived IVs
  // like i+1 or i*4, which we will eventually want to handle.
  if (SCC.size() == 1)
    return nullptr;

  BuiltinInst *FoundBuiltin = nullptr;
  SILArgument *FoundArgument = nullptr;
  IntegerLiteralInst *IncValue = nullptr;
  for (unsigned long i = 0, e = SCC.size(); i != e; ++i) {
    if (auto IV = dyn_cast<SILArgument>(SCC[i])) {
      if (FoundArgument)
        return nullptr;

      FoundArgument = IV;
      continue;
    }

    // TODO: MultiValueInstruction

    auto *I = cast<SILInstruction>(SCC[i]);
    switch (I->getKind()) {
    case SILInstructionKind::BuiltinInst: {
      if (FoundBuiltin)
        return nullptr;

      FoundBuiltin = cast<BuiltinInst>(I);

      SILValue L, R;
      if (!match(FoundBuiltin, m_ApplyInst(BuiltinValueKind::SAddOver,
                                           m_SILValue(L), m_SILValue(R))))
        return nullptr;

      if (match(L, m_IntegerLiteralInst(IncValue)))
        std::swap(L, R);

      if (!match(R, m_IntegerLiteralInst(IncValue)))
        return nullptr;
      break;
    }

    case SILInstructionKind::TupleExtractInst: {
      assert(inSCC(cast<TupleExtractInst>(I)->getOperand(), SCC) &&
             "TupleExtract operand not an induction var");
      break;
    }

    default:
      return nullptr;
    }
  }
  if (!FoundBuiltin || !FoundArgument || !IncValue)
    return nullptr;

  InductionInfoMap[FoundArgument] = IVDesc(FoundBuiltin, IncValue);
  return FoundArgument;
}

void IVInfo::visit(SCCType &SCC) {
  assert(SCC.size() && "SCCs should have an element!!");

  SILArgument *IV;
  if (!(IV = isInductionSequence(SCC)))
    return;

  for (auto node : SCC) {
    if (auto value = dyn_cast<ValueBase>(node))
      InductionVariableMap[value] = IV;
  }
}
