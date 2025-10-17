/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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

//===--- DiagnosticList.h - Diagnostic Definitions --------------*- C++ -*-===//
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
//  This file defines all of the diagnostics emitted by Codira.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_DIAGNOSTICLIST_H
#define LANGUAGE_DIAGNOSTICLIST_H

/// Be *very* careful with what you include here! See include caveats in
/// `ASTBridging.h`.
#include "language/Basic/LanguageBridging.h"
#include <stdint.h>

namespace language {

/// Enumeration describing all of possible diagnostics.
///
/// Each of the diagnostics described in Diagnostics.def has an entry in
/// this enumeration type that uniquely identifies it.
enum class ENUM_EXTENSIBILITY_ATTR(open) DiagID : uint32_t {
#define DIAG(KIND, ID, Group, Options, Text, Signature) ID,
#include "language/AST/DiagnosticsAll.def"
  NumDiagsHandle
};
static_assert(static_cast<uint32_t>(language::DiagID::invalid_diagnostic) == 0,
              "0 is not the invalid diagnostic ID");

constexpr auto NumDiagIDs = static_cast<uint32_t>(DiagID::NumDiagsHandle);

enum class ENUM_EXTENSIBILITY_ATTR(open) FixItID : uint32_t {
#define DIAG(KIND, ID, Group, Options, Text, Signature)
#define FIXIT(ID, Text, Signature) ID,
#include "language/AST/DiagnosticsAll.def"
};

} // end namespace language

#endif /* LANGUAGE_DIAGNOSTICLIST_H */
