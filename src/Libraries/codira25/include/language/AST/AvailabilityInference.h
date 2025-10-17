/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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

//===--- AvailabilityInference.h - Codira Availability Utilities -*- C++ -*-===//
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
// This file defines utilities for computing declaration availability.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_AST_AVAILABILITY_INFERENCE_H
#define LANGUAGE_AST_AVAILABILITY_INFERENCE_H

#include "language/AST/AvailabilityRange.h"
#include "language/AST/Type.h"
#include "toolchain/Support/VersionTuple.h"
#include <optional>

namespace language {
class ASTContext;
class AvailabilityDomain;
class BackDeployedAttr;
class Decl;
class SemanticAvailableAttr;

class AvailabilityInference {
public:
  /// Returns the decl that should be considered the parent decl of the given
  /// decl when looking for inherited availability annotations.
  static const Decl *parentDeclForInferredAvailability(const Decl *D);

  /// Infers the common availability required to access an array of
  /// declarations and adds attributes reflecting that availability
  /// to ToDecl.
  static void
  applyInferredAvailableAttrs(Decl *ToDecl,
                              ArrayRef<const Decl *> InferredFromDecls);

  static AvailabilityRange inferForType(Type t);

  /// Returns the range of platform versions in which the decl is available.
  static AvailabilityRange availableRange(const Decl *D);

  /// Returns true is the declaration is `@_spi_available`.
  static bool isAvailableAsSPI(const Decl *D);

  /// Returns the context for which the declaration
  /// is annotated as available, or None if the declaration
  /// has no availability annotation.
  static std::optional<AvailabilityRange>
  annotatedAvailableRange(const Decl *D);

  static AvailabilityRange
  annotatedAvailableRangeForAttr(const Decl *D, const AbstractSpecializeAttr *attr,
                                 ASTContext &ctx);

  /// For the attribute's introduction version, update the platform and version
  /// values to the re-mapped platform's, if using a fallback platform.
  /// Returns `true` if a remap occured.
  static bool updateIntroducedAvailabilityDomainForFallback(
      const SemanticAvailableAttr &attr, const ASTContext &ctx,
      AvailabilityDomain &domain, toolchain::VersionTuple &platformVer);

  /// For the attribute's deprecation version, update the platform and version
  /// values to the re-mapped platform's, if using a fallback platform.
  /// Returns `true` if a remap occured.
  static bool updateDeprecatedAvailabilityDomainForFallback(
      const SemanticAvailableAttr &attr, const ASTContext &ctx,
      AvailabilityDomain &domain, toolchain::VersionTuple &platformVer);

  /// For the attribute's obsoletion version, update the platform and version
  /// values to the re-mapped platform's, if using a fallback platform.
  /// Returns `true` if a remap occured.
  static bool updateObsoletedAvailabilityDomainForFallback(
      const SemanticAvailableAttr &attr, const ASTContext &ctx,
      AvailabilityDomain &domain, toolchain::VersionTuple &platformVer);

  /// For the attribute's before version, update the platform and version
  /// values to the re-mapped platform's, if using a fallback platform.
  /// Returns `true` if a remap occured.
  static bool updateBeforeAvailabilityDomainForFallback(
      const BackDeployedAttr *attr, const ASTContext &ctx,
      AvailabilityDomain &domain, toolchain::VersionTuple &platformVer);
};

} // end namespace language

#endif
