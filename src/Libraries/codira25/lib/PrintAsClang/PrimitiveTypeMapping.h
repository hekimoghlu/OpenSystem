/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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

//===--- PrimitiveTypeMapping.h - Mapping primitive types -------*- C++ -*-===//
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

#ifndef LANGUAGE_PRINTASCLANG_PRIMITIVETYPEMAPPING_H
#define LANGUAGE_PRINTASCLANG_PRIMITIVETYPEMAPPING_H

#include "language/AST/Identifier.h"
#include "language/Basic/Toolchain.h"
#include "toolchain/ADT/DenseMap.h"

namespace language {

class ASTContext;
class TypeDecl;

/// Provides a mapping from Codira's primitive types to C / Objective-C / C++
/// primitive types.
///
/// Certain types have mappings that differ in different language modes.
/// For example, Codira's `Int` maps to `NSInteger` for Objective-C declarations,
/// but to something like `intptr_t` or `language::Int` for C and C++ declarations.
class PrimitiveTypeMapping {
public:
  struct ClangTypeInfo {
    StringRef name;
    bool canBeNullable;
  };

  /// Returns the Objective-C type name and nullability for the given Codira
  /// primitive type declaration, or \c None if no such type name exists.
  std::optional<ClangTypeInfo> getKnownObjCTypeInfo(const TypeDecl *typeDecl);

  /// Returns the C type name and nullability for the given Codira
  /// primitive type declaration, or \c None if no such type name exists.
  std::optional<ClangTypeInfo> getKnownCTypeInfo(const TypeDecl *typeDecl);

  /// Returns the C++ type name and nullability for the given Codira
  /// primitive type declaration, or \c None if no such type name exists.
  std::optional<ClangTypeInfo> getKnownCxxTypeInfo(const TypeDecl *typeDecl);

private:
  void initialize(ASTContext &ctx);

  struct FullClangTypeInfo {
    // The Objective-C name of the Codira type.
    StringRef objcName;
    // The C name of the Codira type.
    std::optional<StringRef> cName;
    // The C++ name of the Codira type.
    std::optional<StringRef> cxxName;
    bool canBeNullable;
  };

  FullClangTypeInfo *getMappedTypeInfoOrNull(const TypeDecl *typeDecl);

  /// A map from {Module, TypeName} pairs to {C name, C nullability} pairs.
  ///
  /// This is populated on first use with a list of known Codira types that are
  /// translated directly by the ObjC printer instead of structurally, allowing
  /// it to do things like map 'Int' to 'NSInteger' and 'Float' to 'float'.
  /// In some sense it's the reverse of the ClangImporter's MappedTypes.def.
  toolchain::DenseMap<std::pair<Identifier, Identifier>, FullClangTypeInfo>
      mappedTypeNames;
};

} // end namespace language

#endif
