/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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

//===--- CompileTimeInterpolationUtils.cpp -------------------------------===//
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

#include "language/SILOptimizer/Utils/CompileTimeInterpolationUtils.h"
#include "language/AST/ASTContext.h"
#include "language/SIL/SILFunction.h"

using namespace language;

bool language::shouldAttemptEvaluation(SILInstruction *inst) {
  auto *apply = dyn_cast<ApplyInst>(inst);
  if (!apply)
    return true;
  SILFunction *calleeFun = apply->getCalleeFunction();
  if (!calleeFun)
    return false;
  return isConstantEvaluable(calleeFun);
}

std::pair<std::optional<SILBasicBlock::iterator>, std::optional<SymbolicValue>>
language::evaluateOrSkip(ConstExprStepEvaluator &stepEval,
                      SILBasicBlock::iterator instI) {
  SILInstruction *inst = &(*instI);

  // Note that skipping a call conservatively approximates its effects on the
  // interpreter state.
  if (shouldAttemptEvaluation(inst)) {
    return stepEval.tryEvaluateOrElseMakeEffectsNonConstant(instI);
  }
  return stepEval.skipByMakingEffectsNonConstant(instI);
}

void language::getTransitiveUsers(SILInstructionResultArray values,
                               SmallVectorImpl<SILInstruction *> &users) {
  // Collect the instructions that are data dependent on the value using a
  // fix point iteration.
  SmallPtrSet<SILInstruction *, 16> visited;
  SmallVector<SILValue, 16> worklist;
  toolchain::copy(values, std::back_inserter(worklist));
  while (!worklist.empty()) {
    SILValue currVal = worklist.pop_back_val();
    for (Operand *use : currVal->getUses()) {
      SILInstruction *user = use->getUser();
      if (visited.count(user))
        continue;
      visited.insert(user);
      toolchain::copy(user->getResults(), std::back_inserter(worklist));
    }
  }
  users.append(visited.begin(), visited.end());
}
