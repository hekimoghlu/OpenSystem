/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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

//===--- DefaultArgumentKind.h - Default Argument Kind Enum -----*- C++ -*-===//
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
// This file defines the DefaultArgumentKind enumeration.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_DEFAULTARGUMENTKIND_H
#define LANGUAGE_DEFAULTARGUMENTKIND_H

#include "toolchain/ADT/StringRef.h"
#include <cstdint>
#include <string>

namespace toolchain {
class StringRef;
}

namespace language {

/// Describes the kind of default argument a tuple pattern element has.
enum class DefaultArgumentKind : uint8_t {
  /// No default argument.
  None,
  /// A normal default argument.
  Normal,
  /// The default argument is inherited from the corresponding argument of the
  /// overridden declaration.
  Inherited,
  /// The "nil" literal.
  NilLiteral,
  /// An empty array literal.
  EmptyArray,
  /// An empty dictionary literal.
  EmptyDictionary,
  /// A reference to the stored property. This is a special default argument
  /// kind for the synthesized memberwise constructor to emit a call to the
  /// property's initializer.
  StoredProperty,
  // Magic identifier literals expanded at the call site:
#define MAGIC_IDENTIFIER(NAME, STRING) NAME,
#include "language/AST/MagicIdentifierKinds.def"
  /// An expression macro.
  ExpressionMacro
};
enum { NumDefaultArgumentKindBits = 4 };

struct ArgumentAttrs {
  DefaultArgumentKind argumentKind;
  bool isUnavailableInCodira = false;
  toolchain::StringRef CXXOptionsEnumName = "";

  ArgumentAttrs(DefaultArgumentKind argumentKind,
                bool isUnavailableInCodira = false,
                toolchain::StringRef CXXOptionsEnumName = "")
      : argumentKind(argumentKind), isUnavailableInCodira(isUnavailableInCodira),
        CXXOptionsEnumName(CXXOptionsEnumName) {}

  bool operator !=(const DefaultArgumentKind &rhs) const {
    return argumentKind != rhs;
  }

  bool operator==(const DefaultArgumentKind &rhs) const {
    return argumentKind == rhs;
  }

  bool hasDefaultArg() const {
    return argumentKind != DefaultArgumentKind::None;
  }

  bool hasAlternateCXXOptionsEnumName() const {
    return !CXXOptionsEnumName.empty() && isUnavailableInCodira;
  }

  toolchain::StringRef getAlternateCXXOptionsEnumName() const {
    assert(hasAlternateCXXOptionsEnumName() &&
           "Expected a C++ Options type for C++-Interop but found none.");
    return CXXOptionsEnumName;
  }
};

} // end namespace language

#endif // TOOLCHAIN_LANGUAGE_DEFAULTARGUMENTKIND_H

