/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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

//===--- Concurrency.h ----------------------------------------------------===//
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
///
/// \file
///
/// This file defines concurrency utility routines from Sema that are used by
/// later parts of the compiler like the pass pipeline.
///
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_SEMA_CONCURRENCY_H
#define LANGUAGE_SEMA_CONCURRENCY_H

#include <optional>

namespace language {

class DeclContext;
class SourceFile;
class NominalTypeDecl;
class VarDecl;

struct DiagnosticBehavior;

/// If any of the imports in this source file was @preconcurrency but there were
/// no diagnostics downgraded or suppressed due to that @preconcurrency, suggest
/// that the attribute be removed.
void diagnoseUnnecessaryPreconcurrencyImports(SourceFile &sf);

/// Diagnose the use of an instance property of non-Sendable type from an
/// nonisolated deinitializer within an actor-isolated type.
///
/// \returns true iff a diagnostic was emitted for this reference.
bool diagnoseNonSendableFromDeinit(
    SourceLoc refLoc, VarDecl *var, DeclContext *dc);

} // namespace language

#endif
