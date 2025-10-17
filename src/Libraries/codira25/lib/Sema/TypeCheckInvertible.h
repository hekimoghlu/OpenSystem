/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 29, 2024.
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

#ifndef LANGUAGE_SEMA_TYPECHECKINVERTIBLE_H
#define LANGUAGE_SEMA_TYPECHECKINVERTIBLE_H

#include "language/AST/TypeCheckRequests.h"
#include "language/AST/ProtocolConformance.h"

namespace language {

class StorageVisitor {
public:
  /// Visit the instance storage of the given nominal type as seen through
  /// the given declaration context.
  ///
  /// The `this` instance is invoked with each (stored property, property type)
  /// pair for classes/structs and with each (enum elem, associated value type)
  /// pair for enums. It is up to you to implement these handlers by subclassing
  /// this visitor.
  ///
  /// \returns \c true if any call to this \c visitor's handlers returns \c true
  /// and \c false otherwise.
  bool visit(NominalTypeDecl *nominal, DeclContext *dc);

  /// Handle a stored property.
  /// \returns true iff this visitor should stop its walk over the nominal.
  virtual bool operator()(VarDecl *property, Type propertyType) = 0;

  /// Handle an enum associated value.
  /// \returns true iff this visitor should stop its walk over the nominal.
  virtual bool operator()(EnumElementDecl *element, Type elementType) = 0;

  virtual ~StorageVisitor() = default;
};

/// Checks that all stored properties or associated values are Copyable.
void checkCopyableConformance(DeclContext *dc,
                              ProtocolConformanceRef conformance);

/// Checks that all stored properties or associated values are Escapable.
void checkEscapableConformance(DeclContext *dc,
                               ProtocolConformanceRef conformance);
}


#endif // LANGUAGE_SEMA_TYPECHECKINVERTIBLE_H
