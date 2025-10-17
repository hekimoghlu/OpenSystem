/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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

//===---- ConstExtract.h -- Gather Compile-Time-Known Values ----*- C++ -*-===//
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

#ifndef LANGUAGE_CONST_EXTRACT_H
#define LANGUAGE_CONST_EXTRACT_H

#include "language/AST/ConstTypeInfo.h"
#include "toolchain/ADT/ArrayRef.h"
#include <string>
#include <unordered_set>
#include <vector>

namespace toolchain {
class StringRef;
class raw_fd_ostream;
}

namespace language {
class SourceFile;
class DiagnosticEngine;
class ModuleDecl;
} // namespace language

namespace language {
/// Parse a list of string identifiers from a file at the given path,
/// representing names of protocols.
bool
parseProtocolListFromFile(toolchain::StringRef protocolListFilePath,
                          DiagnosticEngine &diags,
                          std::unordered_set<std::string> &protocols);

/// Gather compile-time-known values of properties in nominal type declarations
/// in this file, of types which conform to one of the protocols listed in
/// \c Protocols
std::vector<ConstValueTypeInfo>
gatherConstValuesForPrimary(const std::unordered_set<std::string> &Protocols,
                            const SourceFile *File);

/// Gather compile-time-known values of properties in nominal type declarations
/// in this module, of types which conform to one of the protocols listed in
/// \c Protocols
std::vector<ConstValueTypeInfo>
gatherConstValuesForModule(const std::unordered_set<std::string> &Protocols,
                           ModuleDecl *Module);

/// Serialize a collection of \c ConstValueInfos to JSON at the
/// provided output stream.
bool writeAsJSONToFile(const std::vector<ConstValueTypeInfo> &ConstValueInfos,
                       toolchain::raw_ostream &OS);
} // namespace language

#endif
