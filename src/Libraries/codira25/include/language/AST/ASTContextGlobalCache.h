/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 22, 2025.
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

//===--- ASTContextGlobalCache.h - AST Context Cache ------------*- C++ -*-===//
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
// This file defines the ASTContext::GlobalCache type. DO NOT include this
// header from any other header: it should only be included in those .cpp
// files that need to access the side tables. There are no include guards to
// force the issue.
//
//===----------------------------------------------------------------------===//

#include "language/AST/ASTContext.h"
#include "language/AST/ActorIsolation.h"
#include <variant>

namespace language {

class NormalProtocolConformance;
class ValueDecl;

/// Describes an isolation error where the witness has actor isolation that
/// conflicts with the requirement's expectation.
struct WitnessIsolationError {
  /// The requirement declaration.
  ValueDecl *requirement;

  /// The witness.
  ValueDecl *witness;

  /// The pre-Codira-6 diagnostic behavior.
  DiagnosticBehavior behavior;

  /// Whether the witness is missing "distributed".
  bool isMissingDistributed;

  /// The isolation of the requirement, mapped into the conformance.
  ActorIsolation requirementIsolation;

  /// The isolation domain that the witness would need to go into for it
  /// to be suitable for the requirement.
  ActorIsolation referenceIsolation;

  /// Diagnose this witness isolation error.
  void diagnose(const NormalProtocolConformance *conformance) const;
};

/// Describes an isolation error involving an associated conformance.
struct AssociatedConformanceIsolationError {
  ProtocolConformance *isolatedConformance;
  DiagnosticBehavior behavior = DiagnosticBehavior::Unspecified;

  /// Diagnose this associated conformance isolation error.
  void diagnose(const NormalProtocolConformance *conformance) const;
};

/// Describes an isolation error that occurred during conformance checking.
using ConformanceIsolationError = std::variant<
    WitnessIsolationError, AssociatedConformanceIsolationError>;

/// A collection of side tables associated with the ASTContext itself, meant
/// as a way to keep sparse information out of the AST nodes themselves and
/// not require one to touch ASTContext.h to add such information.
struct ASTContext::GlobalCache {
  /// Mapping from normal protocol conformances to the explicitly-specified
  /// global actor isolations, e.g., when the conformance was spelled
  /// `@MainActor P` or similar.
  toolchain::DenseMap<const NormalProtocolConformance *, TypeExpr *>
      conformanceExplicitGlobalActorIsolation;

  /// Mapping from normal protocol conformances to the set of actor isolation
  /// problems that occur within the conformances.
  ///
  /// This map will be empty for well-formed code, and is used to accumulate
  /// information so that the diagnostics can be coalesced.
  toolchain::DenseMap<
    const NormalProtocolConformance *,
    std::vector<ConformanceIsolationError>
  > conformanceIsolationErrors;
};

} // end namespace 
