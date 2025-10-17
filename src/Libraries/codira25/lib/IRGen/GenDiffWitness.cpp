/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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

//===--- GenDiffWitness.cpp - IRGen for differentiability witnesses -------===//
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
// This file implements IR generation for SIL differentiability witnesses.
//
//===----------------------------------------------------------------------===//

#include "language/AST/PrettyStackTrace.h"
#include "language/Basic/Assertions.h"
#include "language/SIL/SILDifferentiabilityWitness.h"

#include "ConstantBuilder.h"
#include "IRGenModule.h"

using namespace language;
using namespace irgen;

void IRGenModule::emitSILDifferentiabilityWitness(
    SILDifferentiabilityWitness *dw) {
  PrettyStackTraceDifferentiabilityWitness _st(
      "emitting differentiability witness for", dw->getKey());
  // Don't emit declarations.
  if (dw->isDeclaration())
    return;
  // Don't emit `public_external` witnesses.
  if (dw->getLinkage() == SILLinkage::PublicExternal)
    return;
  ConstantInitBuilder builder(*this);
  auto diffWitnessContents = builder.beginStruct();
  assert(dw->getJVP() &&
         "Differentiability witness definition should have JVP");
  assert(dw->getVJP() &&
         "Differentiability witness definition should have VJP");
  diffWitnessContents.add(getAddrOfSILFunction(dw->getJVP(), NotForDefinition));
  diffWitnessContents.add(getAddrOfSILFunction(dw->getVJP(), NotForDefinition));
  getAddrOfDifferentiabilityWitness(
      dw, diffWitnessContents.finishAndCreateFuture());
}
