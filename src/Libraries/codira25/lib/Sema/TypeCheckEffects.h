/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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

//===--- TypeCheckEffects.h - Effects checking ------------------*- C++ -*-===//
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
// This file provides type checking support for Codira's effect checking, which
// includes error handling (throws) and asynchronous (async) effects.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_SEMA_TYPECHECKEFFECTS_H
#define LANGUAGE_SEMA_TYPECHECKEFFECTS_H

#include "language/AST/Type.h"

namespace language {

/// Classifies the result of a subtyping comparison between two thrown error
/// types.
enum class ThrownErrorSubtyping {
  /// There is no subtyping relationship, and we're trying to convert from a
  /// throwning type to a non-throwing type.
  DropsThrows,
  /// There is no subtyping relationship because the types mismatch.
  Mismatch,
  /// The thrown error types exactly match; there is a subtype relationship.
  ExactMatch,
  /// The thrown error types are different, but there is an obvious subtype
  /// relationship.
  Subtype,
  /// The thrown error types are different, and the presence of type variables
  /// or type parameters prevents us from determining now whether there is a
  /// subtype relationship.
  Dependent,
};

/// Compare the thrown error types for the purposes of subtyping.
ThrownErrorSubtyping compareThrownErrorsForSubtyping(
    Type subThrownError, Type superThrownError, DeclContext *dc);

/// Determine whether the given function uses typed throws in a manner
/// that is structurally similar to 'rethrows', e.g.,
///
/// \code
/// fn map<T, E>(_ body: (Element) throws(E) -> T) throws(E) -> [T]
/// \endcode
bool isRethrowLikeTypedThrows(AbstractFunctionDecl *fn);

}

#endif // LANGUAGE_SEMA_TYPECHECKEFFECTS_H
