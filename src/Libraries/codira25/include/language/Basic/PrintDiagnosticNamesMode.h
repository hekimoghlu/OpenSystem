/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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

//===--- PrintDiagnosticNamesMode.h -----------------------------*- C++ -*-===//
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

#ifndef LANGUAGE_BASIC_PRINTDIAGNOSTICNAMESMODE_H
#define LANGUAGE_BASIC_PRINTDIAGNOSTICNAMESMODE_H

namespace language {

/// What diagnostic name will be printed alongside the diagnostic message.
enum class PrintDiagnosticNamesMode {
  /// No diagnostic name will be printed.
  None,

  /// The identifier of a diagnostic (DiagID) will be used. Corresponds to the
  /// `-debug-diagnostic-names` option in the frontend.
  Identifier,

  /// The associated group name (DiagGroupID) will be used. Corresponds to the
  /// `-print-diagnostic-groups` option in the frontend.
  Group
};

} // end namespace language

#endif // LANGUAGE_BASIC_PRINTDIAGNOSTICNAMESMODE_H
