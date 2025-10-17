/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 2, 2024.
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

//===--- DiagnosticsSema.h - Diagnostic Definitions -------------*- C++ -*-===//
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
/// \file
/// This file defines diagnostics for semantic analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_DIAGNOSTICSSEMA_H
#define LANGUAGE_DIAGNOSTICSSEMA_H

#include "language/AST/DiagnosticsCommon.h"

namespace language {
  namespace diag {

    /// Describes the kind of requirement in a protocol.
    enum class RequirementKind : uint8_t {
      Constructor,
      Func,
      Var,
      Subscript
    };

  // Declare common diagnostics objects with their appropriate types.
#define DIAG(KIND, ID, Group, Options, Text, Signature)                    \
      extern detail::DiagWithArguments<void Signature>::type ID;
#define FIXIT(ID, Text, Signature)                                         \
      extern detail::StructuredFixItWithArguments<void Signature>::type ID;
#include "DiagnosticsSema.def"
  }
}

#endif
