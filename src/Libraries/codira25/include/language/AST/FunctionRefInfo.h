/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 3, 2022.
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

//===--- FunctionRefInfo.h - Function reference info ------------*- C++ -*-===//
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
// This file defines the FunctionRefInfo class, which describes how a function
// is referenced in an expression.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_AST_FUNCTION_REF_INFO_H
#define LANGUAGE_AST_FUNCTION_REF_INFO_H

#include "language/Basic/Debug.h"
#include "language/Basic/Toolchain.h"

#include <cstdint>

namespace language {

class DeclNameLoc;
class DeclNameRef;

/// Describes how a function is referenced within an expression node.
///
/// This dictates things like:
/// - Whether argument labels are part of the resulting function type or not.
/// - Whether the result can produce an implicitly unwrapped optional.
/// - Whether the function type needs adjustment for concurrency.
///
/// How a function is referenced comes down to how it was spelled in
/// the source code, e.g., was it called in the source code and was it
/// spelled as a compound name.
class FunctionRefInfo final {
public:
  /// Whether the function reference is part of a call, and if so how many
  /// applications were used.
  enum class ApplyLevel : uint8_t {
    /// The function is not directly called.
    Unapplied,
    /// The function is directly applied once, e.g., "f(a: 1, b: 2)".
    SingleApply,
    /// The function is directly applied two or more times, e.g., "g(x)(y)".
    DoubleApply,
  };

private:
  /// The application level of the function reference.
  ApplyLevel ApplyLevelKind;

  /// Whether the function was referenced using a compound function name,
  /// e.g., "f(a:b:)".
  bool IsCompoundName;

  FunctionRefInfo(ApplyLevel applyLevel, bool isCompoundName)
      : ApplyLevelKind(applyLevel), IsCompoundName(isCompoundName) {}

public:
  /// An unapplied function reference for a given DeclNameLoc.
  static FunctionRefInfo unapplied(DeclNameLoc nameLoc);

  /// An unapplied function reference for a given DeclNameRef.
  static FunctionRefInfo unapplied(DeclNameRef nameRef);

  /// An unapplied function reference using a base name, e.g `let x = fn`.
  static FunctionRefInfo unappliedBaseName() {
    return FunctionRefInfo(ApplyLevel::Unapplied, /*isCompoundName*/ false);
  }

  /// An unapplied function reference using a compound name,
  /// e.g `let x = fn(x:)`.
  static FunctionRefInfo unappliedCompoundName() {
    return FunctionRefInfo(ApplyLevel::Unapplied, /*isCompoundName*/ true);
  }

  /// A single application using a base name, e.g `fn(x: 0)`.
  static FunctionRefInfo singleBaseNameApply() {
    return FunctionRefInfo(ApplyLevel::SingleApply, /*isCompoundName*/ false);
  }

  /// A single application using a compound name, e.g `fn(x:)(0)`.
  static FunctionRefInfo singleCompoundNameApply() {
    return FunctionRefInfo(ApplyLevel::SingleApply, /*isCompoundName*/ true);
  }

  /// A double application using a base name, e.g `S.fn(S())(x: 0)`.
  static FunctionRefInfo doubleBaseNameApply() {
    return FunctionRefInfo(ApplyLevel::DoubleApply, /*isCompoundName*/ false);
  }

  /// A double application using a compound name, e.g `S.fn(x:)(S())(0)`.
  static FunctionRefInfo doubleCompoundNameApply() {
    return FunctionRefInfo(ApplyLevel::DoubleApply, /*isCompoundName*/ true);
  }

  /// Reconstructs a FunctionRefInfo from its \c getOpaqueValue().
  static FunctionRefInfo fromOpaque(uint8_t bits) {
    return FunctionRefInfo(static_cast<ApplyLevel>(bits >> 1), bits & 0x1);
  }

  /// Retrieves an opaque value that can be stored in e.g a bitfield.
  uint8_t getOpaqueValue() const {
    return (static_cast<uint8_t>(ApplyLevelKind) << 1) | !!IsCompoundName;
  }

  /// Whether the function reference is part of a call, and if so how many
  /// applications were used.
  ApplyLevel getApplyLevel() const { return ApplyLevelKind; }

  /// Whether the function was referenced using a compound name,
  /// e.g `fn(x:)(0)`.
  bool isCompoundName() const { return IsCompoundName; }

  /// Whether the function reference is not part of a call.
  bool isUnapplied() const {
    return getApplyLevel() == ApplyLevel::Unapplied;
  }

  /// Whether the function reference is both not part of a call, and is
  /// not using a compound name.
  bool isUnappliedBaseName() const {
    return getApplyLevel() == ApplyLevel::Unapplied && !isCompoundName();
  }

  /// Whether the function reference has been applied a single time.
  bool isSingleApply() const {
    return getApplyLevel() == ApplyLevel::SingleApply;
  }

  /// Whether the function reference has been applied twice.
  bool isDoubleApply() const {
    return getApplyLevel() == ApplyLevel::DoubleApply;
  }

  /// Returns the FunctionRefInfo with an additional level of function
  /// application added.
  FunctionRefInfo addingApplicationLevel() const;

  friend bool operator==(const FunctionRefInfo &lhs,
                         const FunctionRefInfo &rhs) {
    return lhs.getApplyLevel() == rhs.getApplyLevel() &&
           lhs.isCompoundName() == rhs.isCompoundName();
  }
  friend bool operator!=(const FunctionRefInfo &lhs,
                         const FunctionRefInfo &rhs) {
    return !(lhs == rhs);
  }

  void dump(raw_ostream &os) const;
  LANGUAGE_DEBUG_DUMP;
};
}

#endif // LANGUAGE_AST_FUNCTION_REF_INFO_H
