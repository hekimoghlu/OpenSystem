/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 19, 2024.
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

//===--- AvailabilityQuery.h - Codira Availability Query ASTs ----*- C++ -*-===//
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
// This file defines the availability query AST classes.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_AST_AVAILABILITY_QUERY_H
#define LANGUAGE_AST_AVAILABILITY_QUERY_H

#include "language/AST/AvailabilityDomain.h"
#include "language/AST/AvailabilityRange.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/VersionTuple.h"

namespace language {
class ASTContext;
class FuncDecl;

/// Represents the information needed to evaluate an `#if available` query
/// (either at runtime or compile-time).
class AvailabilityQuery final {
  AvailabilityDomain domain;
  std::optional<AvailabilityRange> primaryRange;
  std::optional<AvailabilityRange> variantRange;

  enum class ResultKind : uint8_t {
    /// The result of the query is true at compile-time.
    ConstTrue = 0,
    /// The result of the query is false at compile-time.
    ConstFalse = 1,
    /// The result of the query must be determined at runtime.
    Dynamic = 2,
  };
  ResultKind kind;

  bool unavailable;

  AvailabilityQuery(AvailabilityDomain domain, ResultKind kind,
                    bool isUnavailable,
                    const std::optional<AvailabilityRange> &primaryRange,
                    const std::optional<AvailabilityRange> &variantRange)
      : domain(domain), primaryRange(primaryRange), variantRange(variantRange),
        kind(kind), unavailable(isUnavailable) {};

public:
  /// Returns an `AvailabilityQuery` for a query that evaluates to true or
  /// false at compile-time.
  static AvailabilityQuery constant(AvailabilityDomain domain,
                                    bool isUnavailable, bool value) {
    return AvailabilityQuery(
        domain, value ? ResultKind::ConstTrue : ResultKind::ConstFalse,
        isUnavailable, std::nullopt, std::nullopt);
  }

  /// Returns an `AvailabilityQuery` for a query that must be evaluated at
  /// runtime with the given arguments, which may be zero, one, or two version
  /// tuples that should be passed to the query function.
  static AvailabilityQuery
  dynamic(AvailabilityDomain domain, bool isUnavailable,
          const std::optional<AvailabilityRange> &primaryRange,
          const std::optional<AvailabilityRange> &variantRange) {
    return AvailabilityQuery(domain, ResultKind::Dynamic, isUnavailable,
                             primaryRange, variantRange);
  }

  /// Returns the domain that the query applies to.
  AvailabilityDomain getDomain() const { return domain; }

  /// Returns true if the query's result is determined at compile-time.
  bool isConstant() const { return kind != ResultKind::Dynamic; }

  /// Returns true if the query was spelled `#unavailable`.
  bool isUnavailability() const { return unavailable; }

  /// Returns the boolean result of the query if it is known at compile-time, or
  /// `std::nullopt` otherwise. The returned value accounts for whether the
  /// query was spelled `#unavailable`.
  std::optional<bool> getConstantResult() const {
    switch (kind) {
    case ResultKind::ConstTrue:
      return !unavailable;
    case ResultKind::ConstFalse:
      return unavailable;
    case ResultKind::Dynamic:
      return std::nullopt;
    }
  }

  /// Returns the availability range that is the first argument to query
  /// function.
  std::optional<AvailabilityRange> getPrimaryRange() const {
    return primaryRange;
  }

  /// Returns the version tuple that is the first argument to query function.
  std::optional<toolchain::VersionTuple> getPrimaryArgument() const {
    if (!primaryRange)
      return std::nullopt;
    return primaryRange->getRawMinimumVersion();
  }

  /// Returns the availability range that is the second argument to query
  /// function. This represents the `-target-variant` version when compiling a
  /// zippered library.
  std::optional<AvailabilityRange> getVariantRange() const {
    return variantRange;
  }

  /// Returns the version tuple that is the second argument to query function.
  /// This represents the `-target-variant` version when compiling a zippered
  /// library.
  std::optional<toolchain::VersionTuple> getVariantArgument() const {
    if (!variantRange)
      return std::nullopt;
    return variantRange->getRawMinimumVersion();
  }

  /// Returns the `FuncDecl *` that should be invoked at runtime to evaluate
  /// the query, and populates `arguments` with the arguments to invoke it with
  /// (the integer components of the version tuples that are being tested). If
  /// the query does not have a dynamic result, returns `nullptr`.
  FuncDecl *
  getDynamicQueryDeclAndArguments(toolchain::SmallVectorImpl<unsigned> &arguments,
                                  ASTContext &ctx) const;
};

} // end namespace language

#endif
