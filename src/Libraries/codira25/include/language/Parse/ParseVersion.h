/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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

//===--- ParseVersion.h - Parser Codira Version Numbers ----------*- C++ -*-===//
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

#ifndef LANGUAGE_PARSE_PARSEVERSION_H
#define LANGUAGE_PARSE_PARSEVERSION_H

#include "language/Basic/Version.h"

namespace language {
class DiagnosticEngine;

namespace version {
/// Returns a version from the currently defined LANGUAGE_COMPILER_VERSION.
///
/// If LANGUAGE_COMPILER_VERSION is undefined, this will return the empty
/// compiler version.
Version getCurrentCompilerVersion();
} // namespace version

class VersionParser final {
public:
  /// Parse a version in the form used by the _compiler_version(string-literal)
  /// \#if condition.
  ///
  /// \note This is \em only used for the string literal version, so it includes
  /// backwards-compatibility logic to convert it to something that can be
  /// compared with a modern LANGUAGE_COMPILER_VERSION.
  static std::optional<version::Version>
  parseCompilerVersionString(StringRef VersionString, SourceLoc Loc,
                             DiagnosticEngine *Diags);

  /// Parse a generic version string of the format [0-9]+(.[0-9]+)*
  ///
  /// Version components can be any unsigned 64-bit number.
  static std::optional<version::Version>
  parseVersionString(StringRef VersionString, SourceLoc Loc,
                     DiagnosticEngine *Diags);
};
} // namespace language

#endif // LANGUAGE_PARSE_PARSEVERSION_H
