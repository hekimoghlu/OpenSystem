/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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

//===--- DeadArgumentTransform.cpp ----------------------------------------===//
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

#define DEBUG_TYPE "fso-dead-argument-transform"
#include "FunctionSignatureOpts.h"
#include "language/SIL/DebugUtils.h"
#include "toolchain/Support/CommandLine.h"

using namespace language;

static toolchain::cl::opt<bool> FSODisableDeadArgument(
    "sil-fso-disable-dead-argument",
    toolchain::cl::desc("Do not perform dead argument elimination during FSO. "
                   "Intended only for testing purposes"));

bool FunctionSignatureTransform::DeadArgumentAnalyzeParameters() {
  if (FSODisableDeadArgument)
    return false;

  // Did we decide we should optimize any parameter?
  SILFunction *F = TransformDescriptor.OriginalFunction;

  // If a function has lifetime dependencies, disable FSO's dead param
  // optimization.  Dead params maybe dependency sources and we should not
  // delete them. It is also problematic to dead code params that are not
  // dependency sources, since lifetime dependent sources are stored as indices
  // and deleting dead parameters will require recomputation of these indices.
  if (F->getLoweredFunctionType()->hasLifetimeDependencies()) {
    return false;
  }

  bool SignatureOptimize = false;
  auto Args = F->begin()->getSILFunctionArguments();
  auto OrigShouldModifySelfArgument =
      TransformDescriptor.shouldModifySelfArgument;
  // Analyze the argument information.
  for (unsigned i : indices(Args)) {
    ArgumentDescriptor &A = TransformDescriptor.ArgumentDescList[i];
    if (!A.PInfo.has_value()) {
      // It is not an argument. It could be an indirect result.
      continue;
    }

    if (!A.canOptimizeLiveArg()) {
      continue;
    }

    // Check whether argument is dead.
    if (!hasNonTrivialNonDebugTransitiveUsers(Args[i])) {
      A.IsEntirelyDead = true;
      SignatureOptimize = true;
      if (Args[i]->isSelf())
        TransformDescriptor.shouldModifySelfArgument = true;
    }
  }

  if (F->getLoweredFunctionType()->isPolymorphic()) {
    // If the set of dead arguments contains only type arguments,
    // don't remove them, because it would produce a slower code
    // for generic functions.
    bool HasNonTypeDeadArguments = false;
    for (auto &AD : TransformDescriptor.ArgumentDescList) {
      if (AD.IsEntirelyDead &&
          !isa<AnyMetatypeType>(AD.Arg->getType().getASTType())) {
        HasNonTypeDeadArguments = true;
        break;
      }
    }

    if (!HasNonTypeDeadArguments) {
      for (auto &AD : TransformDescriptor.ArgumentDescList) {
        if (AD.IsEntirelyDead) {
          AD.IsEntirelyDead = false;
        }
      }
      TransformDescriptor.shouldModifySelfArgument =
          OrigShouldModifySelfArgument;
      SignatureOptimize = false;
    }
  }

  return SignatureOptimize;
}

void FunctionSignatureTransform::DeadArgumentTransformFunction() {
  SILFunction *F = TransformDescriptor.OriginalFunction;
  SILBasicBlock *BB = &*F->begin();
  for (const ArgumentDescriptor &AD : TransformDescriptor.ArgumentDescList) {
    if (!AD.IsEntirelyDead)
      continue;
    eraseUsesOfValue(BB->getArgument(AD.Index));
  }
}

void FunctionSignatureTransform::DeadArgumentFinalizeOptimizedFunction() {
  auto *BB = &*TransformDescriptor.OptimizedFunction.get()->begin();
  // Remove any dead argument starting from the last argument to the first.
  for (ArgumentDescriptor &AD : reverse(TransformDescriptor.ArgumentDescList)) {
    if (!AD.IsEntirelyDead)
      continue;
    AD.WasErased = true;
    BB->eraseArgument(AD.Arg->getIndex());
  }
}
