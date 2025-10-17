/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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

//===--- DiagnosticList.cpp - Diagnostic Definitions ----------------------===//
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

#include "language/AST/DiagnosticList.h"
#include "language/AST/DiagnosticsCommon.h"
#include "language/Basic/Assertions.h"
using namespace language;

// Define all of the diagnostic objects and initialize them with their 
// diagnostic IDs.
namespace language {
  namespace diag {
#define DIAG(KIND, ID, Group, Options, Text, Signature)                      \
    detail::DiagWithArguments<void Signature>::type ID = {DiagID::ID};
#define FIXIT(ID, Text, Signature) \
    detail::StructuredFixItWithArguments<void Signature>::type ID = {FixItID::ID};
#include "language/AST/DiagnosticsAll.def"
  } // end namespace diag
} // end namespace language
