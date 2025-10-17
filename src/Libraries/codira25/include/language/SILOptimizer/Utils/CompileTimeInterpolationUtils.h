/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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

//===--- CompileTimeInterpolationUtils.h ----------------------------------===//
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

// Utilities for the compile-time string interpolation approach used by the
// OSLogOptimization pass.

#ifndef LANGUAGE_SILOPTIMIZER_COMPILE_TIME_INTERPOLATION_H
#define LANGUAGE_SILOPTIMIZER_COMPILE_TIME_INTERPOLATION_H

#include "language/SIL/SILBasicBlock.h"
#include "language/SIL/SILConstants.h"
#include "language/SILOptimizer/Utils/ConstExpr.h"

namespace language {

/// Decide if the given instruction (which could possibly be a call) should
/// be constant evaluated.
///
/// \returns true iff the given instruction is not a call or if it is, it calls
/// a known constant-evaluable function such as string append etc., or calls
/// a function annotate as "constant_evaluable".
bool shouldAttemptEvaluation(SILInstruction *inst);

/// Skip or evaluate the given instruction based on the evaluation policy and
/// handle errors. The policy is to evaluate all non-apply instructions as well
/// as apply instructions that are marked as "constant_evaluable".
std::pair<std::optional<SILBasicBlock::iterator>, std::optional<SymbolicValue>>
evaluateOrSkip(ConstExprStepEvaluator &stepEval, SILBasicBlock::iterator instI);

/// Given a vector of SILValues \p worklist, compute the set of transitive
/// users of these values (excluding the worklist values) by following the
/// use-def chain starting at value. Note that this function does not follow
/// use-def chains though branches.
void getTransitiveUsers(SILInstructionResultArray values,
                        SmallVectorImpl<SILInstruction *> &users);
} // end namespace language
#endif
