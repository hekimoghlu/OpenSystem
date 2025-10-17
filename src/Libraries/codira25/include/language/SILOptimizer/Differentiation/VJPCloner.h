/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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

//===--- VJPCloner.h - VJP function generation ----------------*- C++ -*---===//
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
//
// This file defines a helper class for generating VJP functions for automatic
// differentiation.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_SILOPTIMIZER_UTILS_DIFFERENTIATION_VJPCLONER_H
#define LANGUAGE_SILOPTIMIZER_UTILS_DIFFERENTIATION_VJPCLONER_H

#include "language/SILOptimizer/Analysis/DifferentiableActivityAnalysis.h"
#include "language/SILOptimizer/Differentiation/DifferentiationInvoker.h"
#include "language/SILOptimizer/Differentiation/LinearMapInfo.h"
#include "language/SIL/LoopInfo.h"

namespace language {
namespace autodiff {

class ADContext;
class PullbackCloner;

/// A helper class for generating VJP functions.
class VJPCloner final {
  class Implementation;
  Implementation &impl;

public:
  /// Creates a VJP cloner.
  ///
  /// The parent VJP cloner stores the original function and an empty
  /// to-be-generated pullback function.
  explicit VJPCloner(ADContext &context, SILDifferentiabilityWitness *witness,
                     SILFunction *vjp, DifferentiationInvoker invoker);
  ~VJPCloner();

  ADContext &getContext() const;
  SILModule &getModule() const;
  SILFunction &getOriginal() const;
  SILFunction &getVJP() const;
  SILFunction &getPullback() const;
  SILDifferentiabilityWitness *getWitness() const;
  const AutoDiffConfig &getConfig() const;
  DifferentiationInvoker getInvoker() const;
  LinearMapInfo &getPullbackInfo() const;
  SILLoopInfo *getLoopInfo() const;
  const DifferentiableActivityInfo &getActivityInfo() const;

  /// Performs VJP generation on the empty VJP function. Returns true if any
  /// error occurs.
  bool run();
};

} // end namespace autodiff
} // end namespace language

#endif // LANGUAGE_SILOPTIMIZER_UTILS_DIFFERENTIATION_VJPCLONER_H
