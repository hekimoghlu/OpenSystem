/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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

//===--- SymbolGraphOptions.h - Codira SymbolGraph Options -----------------===//
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

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/DenseSet.h"
#include "toolchain/TargetParser/Triple.h"

#include "language/AST/AttrKind.h"

#ifndef LANGUAGE_SYMBOLGRAPHGEN_SYMBOLGRAPHOPTIONS_H
#define LANGUAGE_SYMBOLGRAPHGEN_SYMBOLGRAPHOPTIONS_H

namespace language {
namespace symbolgraphgen {

struct SymbolGraphOptions {
  /// The directory to output the symbol graph JSON files.
  StringRef OutputDir = {};

  /// The target of the module.
  toolchain::Triple Target = {};
  /// Pretty-print the JSON with newlines and indentation.
  bool PrettyPrint = false;

  /// The minimum access level that symbols must have in order to be
  /// included in the graph.
  AccessLevel MinimumAccessLevel = AccessLevel::Public;

  /// Emit members gotten through class inheritance or protocol default
  /// implementations with compound, "SYNTHESIZED" USRs.
  bool EmitSynthesizedMembers = false;
  
  /// Whether to print informational messages when rendering
  /// a symbol graph.
  bool PrintMessages = false;
  
  /// Whether to skip docs for symbols with compound, "SYNTHESIZED" USRs.
  bool SkipInheritedDocs = false;

  /// Whether to skip emitting symbols that are implementations of protocol requirements or
  /// inherited from protocol extensions.
  bool SkipProtocolImplementations = false;

  /// Whether to emit symbols with SPI information.
  bool IncludeSPISymbols = false;

  /// Whether to include documentation for clang nodes or not.
  bool IncludeClangDocs = false;

  /// Whether to emit "language.extension" symbols for extensions to external types
  /// along with "extensionTo" relationships instead of directly associating
  /// members and conformances with the extended nominal.
  bool EmitExtensionBlockSymbols = false;

  /// Whether to print information for private symbols in system modules.
  /// This should be left as `false` when printing a full-module symbol graph,
  /// but SourceKit should be able to load the information when pulling symbol
  /// information for individual queries.
  bool PrintPrivateSystemSymbols = false;

  /// If this has a value specifies an explicit allow list of reexported module
  /// names that should be included symbol graph.
  std::optional<toolchain::ArrayRef<StringRef>> AllowedReexportedModules = {};

  /// If set, a list of availability platforms to restrict (or block) when
  /// rendering symbol graphs.
  std::optional<toolchain::DenseSet<StringRef>> AvailabilityPlatforms = {};

  /// Whether `AvailabilityPlatforms` is an allow list or a block list.
  bool AvailabilityIsBlockList = false;
};

} // end namespace symbolgraphgen
} // end namespace language

#endif // LANGUAGE_SYMBOLGRAPHGEN_SYMBOLGRAPHOPTIONS_H
