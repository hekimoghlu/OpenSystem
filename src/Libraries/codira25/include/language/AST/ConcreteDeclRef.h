/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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

//===--- ConcreteDeclRef.h - Reference to a concrete decl -------*- C++ -*-===//
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
// This file defines the ConcreteDeclRef class, which provides a reference to
// a declaration that is potentially specialized.
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_AST_CONCRETEDECLREF_H
#define LANGUAGE_AST_CONCRETEDECLREF_H

#include "language/Basic/Debug.h"
#include "language/Basic/Toolchain.h"
#include "language/AST/SubstitutionMap.h"
#include "language/AST/TypeAlignments.h"
#include "toolchain/ADT/PointerUnion.h"
#include "toolchain/Support/Compiler.h"
#include <cstring>

namespace language {

class ASTContext;
class SourceManager;
class ValueDecl;

/// A reference to a concrete representation of a particular declaration,
/// providing substitutions for all type parameters of the original,
/// underlying declaration.
class ConcreteDeclRef {
  /// The declaration.
  ValueDecl *decl = nullptr;

  /// The substitutions.
  SubstitutionMap substitutions;

public:
  /// Create an empty declaration reference.
  ConcreteDeclRef() { }

  /// Construct a reference to the given value.
  ConcreteDeclRef(ValueDecl *decl) : decl(decl) { }

  /// Construct a reference to the given value, specialized with the given
  /// substitutions.
  ///
  /// \param decl The declaration to which this reference refers, which will
  /// be specialized by applying the given substitutions.
  ///
  /// \param substitutions The complete set of substitutions to apply to the
  /// given declaration.
  ConcreteDeclRef(ValueDecl *decl, SubstitutionMap substitutions)
    : decl(decl), substitutions(substitutions) { }

  /// Determine whether this declaration reference refers to anything.
  explicit operator bool() const { return decl != nullptr; }

  /// Retrieve the declarations to which this reference refers.
  ValueDecl *getDecl() const { return decl; }

  /// Retrieve a reference to the declaration this one overrides.
  ConcreteDeclRef getOverriddenDecl() const;

  /// Retrieve a reference to the given declaration that this one overrides.
  ConcreteDeclRef getOverriddenDecl(ValueDecl *overriddenDecl) const;

  /// Determine whether this reference specializes the declaration to which
  /// it refers.
  bool isSpecialized() const { return !substitutions.empty(); }

  /// For a specialized reference, return the set of substitutions applied to
  /// the declaration reference.
  SubstitutionMap getSubstitutions() const { return substitutions; }

  friend bool operator==(ConcreteDeclRef lhs, ConcreteDeclRef rhs) {
    return lhs.decl == rhs.decl && lhs.substitutions == rhs.substitutions;
  }
  
  /// Dump a debug representation of this reference.
  void dump(raw_ostream &os) const;
  LANGUAGE_DEBUG_DUMP;
};

} // end namespace language

#endif
