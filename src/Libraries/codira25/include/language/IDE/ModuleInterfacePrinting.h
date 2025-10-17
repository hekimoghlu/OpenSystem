/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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

//===--- ModuleInterfacePrinting.h - Routines to print module interface ---===//
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

#ifndef LANGUAGE_IDE_MODULE_INTERFACE_PRINTING_H
#define LANGUAGE_IDE_MODULE_INTERFACE_PRINTING_H

#include "language/Basic/Toolchain.h"
#include "language/Basic/OptionSet.h"

#include <optional>

#include <string>
#include <vector>

namespace language::Core {
class Module;
}

namespace language {
class ASTContext;
class ASTPrinter;
class ModuleDecl;
class SourceFile;
class Type;
struct PrintOptions;

namespace ide {

/// Flags used when traversing a module for printing.
enum class ModuleTraversal : unsigned {
  /// Visit modules even if their contents wouldn't be visible to name lookup.
  VisitHidden = 0x01,
  /// Visit submodules.
  VisitSubmodules = 0x02,
  /// Skip the declarations in a Codira overlay module.
  SkipOverlay = 0x04,
  /// Visit exported modules where their public module name matches the current
  /// module.
  VisitMatchingExported = 0x08,
};

/// Options used to describe the traversal of a module for printing.
using ModuleTraversalOptions = OptionSet<ModuleTraversal>;

void collectModuleGroups(ModuleDecl *M, SmallVectorImpl<StringRef> &Into);

std::optional<StringRef> findGroupNameForUSR(ModuleDecl *M, StringRef USR);

bool printTypeInterface(ModuleDecl *M, Type Ty, ASTPrinter &Printer,
                        std::string &TypeName, std::string &Error);

bool printTypeInterface(ModuleDecl *M, StringRef TypeUSR, ASTPrinter &Printer,
                        std::string &TypeName, std::string &Error);

void printModuleInterface(ModuleDecl *M, ArrayRef<StringRef> GroupNames,
                          ModuleTraversalOptions TraversalOptions,
                          ASTPrinter &Printer, const PrintOptions &Options,
                          const bool PrintSynthesizedExtensions);

/// Print the interface for a header that has been imported via the implicit
/// objc header importing feature.
void printHeaderInterface(StringRef Filename, ASTContext &Ctx,
                          ASTPrinter &Printer, const PrintOptions &Options);


/// Print the interface for a given language source file.
void printCodiraSourceInterface(SourceFile &File, ASTPrinter &Printer,
                               const PrintOptions &Options);

/// Print the symbolic Codira interface for a given imported clang module.
void printSymbolicCodiraClangModuleInterface(ModuleDecl *M, ASTPrinter &Printer,
                                            const language::Core::Module *clangModule);

} // namespace ide

} // namespace language

#endif // LANGUAGE_IDE_MODULE_INTERFACE_PRINTING_H
