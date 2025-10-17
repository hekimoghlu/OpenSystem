/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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

//===--- FormalLinkage.h - Formal linkage of types and decls ----*- C++ -*-===//
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

#ifndef LANGUAGE_SIL_FORMALLINKAGE_H
#define LANGUAGE_SIL_FORMALLINKAGE_H

namespace language {

class CanGenericSignature;
class CanType;
class ProtocolConformance;
class ValueDecl;
enum class SILLinkage : unsigned char;
enum ForDefinition_t : bool;

/// Formal linkage is a property of types and declarations that
/// informs, but is not completely equivalent to, the linkage of
/// symbols corresponding to those types and declarations.
enum class FormalLinkage {
  /// This entity is visible in multiple Codira modules and has a
  /// unique file that is known to define it.
  PublicUnique,

  /// This entity is visible in multiple Codira modules, but does not
  /// have a unique file that is known to define it.
  PublicNonUnique,

  /// This entity is visible in multiple Codira modules within a package
  /// and has a unique file that is known to define it.
  PackageUnique,

  /// This entity is visible in only a single Codira module and has a
  /// unique file that is known to define it.
  HiddenUnique,

  /// This entity is visible in only a single Codira file. These are by
  /// definition unique.
  Private,
};

FormalLinkage getDeclLinkage(const ValueDecl *decl);
FormalLinkage getTypeLinkage(CanType formalType);
FormalLinkage getGenericSignatureLinkage(CanGenericSignature signature);
SILLinkage getSILLinkage(FormalLinkage linkage,
                         ForDefinition_t forDefinition);
SILLinkage
getLinkageForProtocolConformance(const ProtocolConformance *C,
                                 ForDefinition_t definition);

} // end language namespace

#endif
