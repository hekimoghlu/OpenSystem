/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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

//===--- RequirementSignature.h - Requirement Signature AST -----*- C++ -*-===//
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
// This file defines the RequirementSignature class.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_AST_REQUIREMENT_SIGNATURE_H
#define LANGUAGE_AST_REQUIREMENT_SIGNATURE_H

#include "language/AST/GenericSignature.h"
#include "language/AST/Type.h"

namespace language {

/// A description of a typealias defined in a protocol.
class ProtocolTypeAlias final {
  Identifier Name;
  Type UnderlyingType;

public:
  ProtocolTypeAlias(Identifier name, Type underlyingType)
    : Name(name), UnderlyingType(underlyingType) {}

  /// Returns the name of the typealias.
  Identifier getName() const { return Name; }

  /// Returns the underlying type of the typealias.
  Type getUnderlyingType() const { return UnderlyingType; }
};

/// The requirements that describe a protocol from the viewpoint of the
/// generics system.
class RequirementSignature final {
  ArrayRef<Requirement> Requirements;
  ArrayRef<ProtocolTypeAlias> TypeAliases;
  GenericSignatureErrors Errors;

public:
  RequirementSignature(GenericSignatureErrors errors = GenericSignatureErrors())
    : Errors(errors) {}

  RequirementSignature(ArrayRef<Requirement> requirements,
                       ArrayRef<ProtocolTypeAlias> typeAliases,
                       GenericSignatureErrors errors = GenericSignatureErrors())
    : Requirements(requirements), TypeAliases(typeAliases), Errors(errors) {}

  /// The requirements including any inherited protocols and conformances for
  /// associated types that are introduced in this protocol.
  ///
  /// Requirements implied via any other protocol (e.g., inherited protocols
  /// of the inherited protocols) are not mentioned.
  ///
  /// The conformance requirements listed here become entries in witness tables
  /// for conformances to this protocol.
  ArrayRef<Requirement> getRequirements() const {
    return Requirements;
  }

  ArrayRef<ProtocolTypeAlias> getTypeAliases() const {
    return TypeAliases;
  }

  GenericSignatureErrors getErrors() const {
    return Errors;
  }

  void getRequirementsWithInverses(
      ProtocolDecl *owner,
      SmallVector<Requirement, 2> &reqs,
      SmallVector<InverseRequirement, 2> &inverses) const;

  void print(ProtocolDecl *owner, raw_ostream &OS,
             const PrintOptions &Options = PrintOptions()) const;
  void print(ProtocolDecl *owner, ASTPrinter &Printer,
             const PrintOptions &Opts = PrintOptions()) const;

  static RequirementSignature getPlaceholderRequirementSignature(
      const ProtocolDecl *proto, GenericSignatureErrors errors);
};

} // end namespace language

#endif // LANGUAGE_AST_REQUIREMENT_SIGNATURE_H
