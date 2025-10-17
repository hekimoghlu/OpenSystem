/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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

//===--- TypeLoc.h - Codira Language Type Locations --------------*- C++ -*-===//
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
// This file defines the TypeLoc struct and related structs.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_TYPELOC_H
#define LANGUAGE_TYPELOC_H

#include "language/Basic/SourceLoc.h"
#include "language/AST/Type.h"
#include "language/AST/TypeAlignments.h"
#include "toolchain/ADT/PointerIntPair.h"

namespace language {

class ASTContext;
class TypeRepr;

/// TypeLoc - Provides source location information for a parsed type.
/// A TypeLoc is stored in AST nodes which use an explicitly written type.
class alignas(1 << TypeReprAlignInBits) TypeLoc {
  Type Ty;
  TypeRepr *TyR = nullptr;

public:
  TypeLoc() {}
  TypeLoc(TypeRepr *TyR) : TyR(TyR) {}
  TypeLoc(TypeRepr *TyR, Type Ty) : TyR(TyR) {
    setType(Ty);
  }

  bool wasValidated() const { return !Ty.isNull(); }
  bool isError() const;

  // FIXME: We generally shouldn't need to build TypeLocs without a location.
  static TypeLoc withoutLoc(Type T) {
    TypeLoc result;
    result.Ty = T;
    return result;
  }

  /// Get the representative location of this type, for diagnostic
  /// purposes.
  /// This location is not necessarily the start location of the type repr.
  SourceLoc getLoc() const;
  SourceRange getSourceRange() const;

  bool hasLocation() const { return TyR != nullptr; }
  TypeRepr *getTypeRepr() const { return TyR; }
  Type getType() const { return Ty; }

  bool isNull() const { return getType().isNull() && TyR == nullptr; }

  void setType(Type Ty);

  friend toolchain::hash_code hash_value(const TypeLoc &owner) {
    return toolchain::hash_combine(owner.Ty.getPointer(), owner.TyR);
  }

  friend bool operator==(const TypeLoc &lhs,
                         const TypeLoc &rhs) {
    return lhs.Ty.getPointer() == rhs.Ty.getPointer()
        && lhs.TyR == rhs.TyR;
  }

  friend bool operator!=(const TypeLoc &lhs,
                         const TypeLoc &rhs) {
    return !(lhs == rhs);
  }
};

} // end namespace toolchain

#endif
