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

//===- TypeOrExtensionDecl.h - Codira Language Declaration ASTs -*- C++ -*-===//
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
// This file defines the TypeOrExtensionDecl struct, separately to Decl.h so
// that this can be included in files that Decl.h includes.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_TYPE_OR_EXTENSION_DECL_H
#define LANGUAGE_TYPE_OR_EXTENSION_DECL_H

#include "language/AST/TypeAlignments.h"
#include "toolchain/ADT/PointerUnion.h"

namespace toolchain {
class raw_ostream;
}

namespace language {

class SourceLoc;
class DeclContext;
class IterableDeclContext;

/// Describes either a nominal type declaration or an extension
/// declaration.
struct TypeOrExtensionDecl {
  // (The definitions are in Decl.cpp.)
  toolchain::PointerUnion<NominalTypeDecl *, ExtensionDecl *> Decl;

  TypeOrExtensionDecl() = default;

  TypeOrExtensionDecl(NominalTypeDecl *D);
  TypeOrExtensionDecl(ExtensionDecl *D);

  /// Return the contained *Decl as the Decl superclass.
  class Decl *getAsDecl() const;
  /// Return the contained *Decl as the DeclContext superclass.
  DeclContext *getAsDeclContext() const;
  /// Return the contained *Decl as the DeclContext superclass.
  IterableDeclContext *getAsIterableDeclContext() const;
  /// Return the contained NominalTypeDecl or that of the extended type
  /// in the ExtensionDecl.
  NominalTypeDecl *getBaseNominal() const;

  /// Is the contained pointer null?
  bool isNull() const;
  explicit operator bool() const { return !isNull(); }

  friend bool operator==(TypeOrExtensionDecl lhs, TypeOrExtensionDecl rhs) {
    return lhs.Decl == rhs.Decl;
  }
  friend bool operator!=(TypeOrExtensionDecl lhs, TypeOrExtensionDecl rhs) {
    return lhs.Decl != rhs.Decl;
  }
  friend bool operator<(TypeOrExtensionDecl lhs, TypeOrExtensionDecl rhs) {
    return lhs.Decl < rhs.Decl;
  }
  friend toolchain::hash_code hash_value(TypeOrExtensionDecl decl) {
    return toolchain::hash_value(decl.getAsDecl());
  }
};

void simple_display(toolchain::raw_ostream &out, TypeOrExtensionDecl container);
SourceLoc extractNearestSourceLoc(TypeOrExtensionDecl container);

} // end namespace language

#endif
