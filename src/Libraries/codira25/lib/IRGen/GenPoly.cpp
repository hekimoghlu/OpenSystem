/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 23, 2025.
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

//===--- GenPoly.cpp - Codira IR Generation for Polymorphism ---------------===//
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
//  This file implements IR generation for polymorphic operations in Codira.
//
//===----------------------------------------------------------------------===//

#include "language/AST/ASTContext.h"
#include "language/AST/ASTVisitor.h"
#include "language/AST/Types.h"
#include "language/AST/Decl.h"
#include "language/AST/GenericEnvironment.h"
#include "language/Basic/Assertions.h"
#include "language/SIL/SILInstruction.h"
#include "language/SIL/SILModule.h"
#include "language/SIL/SILType.h"
#include "toolchain/IR/DerivedTypes.h"

#include "Explosion.h"
#include "IRGenFunction.h"
#include "IRGenModule.h"
#include "LoadableTypeInfo.h"
#include "TypeVisitor.h"
#include "GenTuple.h"
#include "GenPoly.h"
#include "GenType.h"

using namespace language;
using namespace irgen;

static SILType applyPrimaryArchetypes(IRGenFunction &IGF,
                                      SILType type) {
  if (!type.hasTypeParameter()) {
    return type;
  }

  auto substType =
    IGF.IGM.getGenericEnvironment()->mapTypeIntoContext(type.getASTType())
      ->getCanonicalType();
  return SILType::getPrimitiveType(substType, type.getCategory());
}

/// Given a substituted explosion, re-emit it as an unsubstituted one.
///
/// For example, given an explosion which begins with the
/// representation of an (Int, Float), consume that and produce the
/// representation of an (Int, T).
///
/// The substitutions must carry origTy to substTy.
void irgen::reemitAsUnsubstituted(IRGenFunction &IGF,
                                  SILType expectedTy, SILType substTy,
                                  Explosion &in, Explosion &out) {
  expectedTy = applyPrimaryArchetypes(IGF, expectedTy);

  ExplosionSchema expectedSchema;
  cast<LoadableTypeInfo>(IGF.IGM.getTypeInfo(expectedTy))
    .getSchema(expectedSchema);

#ifndef NDEBUG
  auto &substTI = IGF.IGM.getTypeInfo(applyPrimaryArchetypes(IGF, substTy));
  assert(expectedSchema.size() ==
         cast<LoadableTypeInfo>(substTI).getExplosionSize());
#endif

  for (ExplosionSchema::Element &elt : expectedSchema) {
    toolchain::Value *value = in.claimNext();
    assert(elt.isScalar());

    // The only type differences we expect here should be due to
    // substitution of class archetypes.
    if (value->getType() != elt.getScalarType()) {
      value = IGF.Builder.CreateBitCast(value, elt.getScalarType(),
                                        value->getName() + ".asUnsubstituted");
    }
    out.add(value);
  }
}
