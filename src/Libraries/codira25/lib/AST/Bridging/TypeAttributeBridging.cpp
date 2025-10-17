/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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

//===--- Bridging/TypeAttributeBridging.cpp -------------------------------===//
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

#include "language/AST/ASTBridging.h"

#include "language/AST/ASTContext.h"
#include "language/AST/Attr.h"
#include "language/AST/Identifier.h"
#include "language/Basic/Assertions.h"

using namespace language;

//===----------------------------------------------------------------------===//
// MARK: TypeAttributes
//===----------------------------------------------------------------------===//

// Define `.asTypeAttr` on each BridgedXXXTypeAttr type.
#define SIMPLE_TYPE_ATTR(...)
#define TYPE_ATTR(SPELLING, CLASS)                                             \
  LANGUAGE_NAME("getter:Bridged" #CLASS "TypeAttr.asTypeAttribute(self:)")        \
  BridgedTypeAttribute Bridged##CLASS##TypeAttr_asTypeAttribute(               \
      Bridged##CLASS##TypeAttr attr) {                                         \
    return attr.unbridged();                                                   \
  }
#include "language/AST/TypeAttr.def"

BridgedOptionalTypeAttrKind
BridgedOptionalTypeAttrKind_fromString(BridgedStringRef cStr) {
  auto optKind = TypeAttribute::getAttrKindFromString(cStr.unbridged());
  if (!optKind) {
    return BridgedOptionalTypeAttrKind();
  }
  return *optKind;
}

BridgedTypeAttribute BridgedTypeAttribute_createSimple(
    BridgedASTContext cContext, language::TypeAttrKind kind,
    BridgedSourceLoc cAtLoc, BridgedSourceLoc cNameLoc) {
  return TypeAttribute::createSimple(cContext.unbridged(), kind,
                                     cAtLoc.unbridged(), cNameLoc.unbridged());
}

BridgedConventionTypeAttr BridgedConventionTypeAttr_createParsed(
    BridgedASTContext cContext, BridgedSourceLoc cAtLoc,
    BridgedSourceLoc cKwLoc, BridgedSourceRange cParens, BridgedStringRef cName,
    BridgedSourceLoc cNameLoc, BridgedDeclNameRef cWitnessMethodProtocol,
    BridgedStringRef cClangType, BridgedSourceLoc cClangTypeLoc) {
  return new (cContext.unbridged()) ConventionTypeAttr(
      cAtLoc.unbridged(), cKwLoc.unbridged(), cParens.unbridged(),
      {cName.unbridged(), cNameLoc.unbridged()},
      cWitnessMethodProtocol.unbridged(),
      {cClangType.unbridged(), cClangTypeLoc.unbridged()});
}

BridgedDifferentiableTypeAttr BridgedDifferentiableTypeAttr_createParsed(
    BridgedASTContext cContext, BridgedSourceLoc cAtLoc,
    BridgedSourceLoc cNameLoc, BridgedSourceRange cParensRange,
    BridgedDifferentiabilityKind cKind, BridgedSourceLoc cKindLoc) {
  return new (cContext.unbridged()) DifferentiableTypeAttr(
      cAtLoc.unbridged(), cNameLoc.unbridged(), cParensRange.unbridged(),
      {unbridged(cKind), cKindLoc.unbridged()});
}

BridgedIsolatedTypeAttr BridgedIsolatedTypeAttr_createParsed(
    BridgedASTContext cContext, BridgedSourceLoc cAtLoc,
    BridgedSourceLoc cNameLoc, BridgedSourceRange cParensRange,

    BridgedIsolatedTypeAttrIsolationKind cIsolation,
    BridgedSourceLoc cIsolationLoc) {
  auto isolationKind = [=] {
    switch (cIsolation) {
    case BridgedIsolatedTypeAttrIsolationKind_DynamicIsolation:
      return IsolatedTypeAttr::IsolationKind::Dynamic;
    }
    toolchain_unreachable("bad kind");
  }();
  return new (cContext.unbridged()) IsolatedTypeAttr(
      cAtLoc.unbridged(), cNameLoc.unbridged(), cParensRange.unbridged(),
      {isolationKind, cIsolationLoc.unbridged()});
}

BridgedOpaqueReturnTypeOfTypeAttr
BridgedOpaqueReturnTypeOfTypeAttr_createParsed(
    BridgedASTContext cContext, BridgedSourceLoc cAtLoc,
    BridgedSourceLoc cKwLoc, BridgedSourceRange cParens,
    BridgedStringRef cMangled, BridgedSourceLoc cMangledLoc, size_t index,
    BridgedSourceLoc cIndexLoc) {
  return new (cContext.unbridged()) OpaqueReturnTypeOfTypeAttr(
      cAtLoc.unbridged(), cKwLoc.unbridged(), cParens.unbridged(),
      {cMangled.unbridged(), cMangledLoc.unbridged()},
      {static_cast<unsigned int>(index), cIndexLoc.unbridged()});
}
