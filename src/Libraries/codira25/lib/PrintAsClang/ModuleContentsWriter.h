/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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

//===--- ModuleContentsWriter.h - Walk module to print ObjC/C++ -*- C++ -*-===//
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

#ifndef LANGUAGE_PRINTASCLANG_MODULECONTENTSWRITER_H
#define LANGUAGE_PRINTASCLANG_MODULECONTENTSWRITER_H

#include "language/AST/AttrKind.h"
#include "language/Basic/Toolchain.h"
#include "toolchain/ADT/PointerUnion.h"
#include "toolchain/ADT/SmallPtrSet.h"
#include "toolchain/ADT/StringSet.h"

namespace language::Core {
  class Module;
}

namespace language {
class Decl;
class ModuleDecl;
class CodiraToClangInteropContext;

using ImportModuleTy = PointerUnion<ModuleDecl*, const language::Core::Module*>;

/// Prints the declarations of \p M to \p os and collecting imports in
/// \p imports along the way.
void printModuleContentsAsObjC(raw_ostream &os,
                               toolchain::SmallPtrSetImpl<ImportModuleTy> &imports,
                               ModuleDecl &M,
                               CodiraToClangInteropContext &interopContext);

void printModuleContentsAsC(raw_ostream &os,
                            toolchain::SmallPtrSetImpl<ImportModuleTy> &imports,
                            ModuleDecl &M,
                            CodiraToClangInteropContext &interopContext);

struct EmittedClangHeaderDependencyInfo {
    /// The set of imported modules used by this module.
    SmallPtrSet<ImportModuleTy, 8> imports;
    /// True if the printed module depends on types from the Stdlib module.
    bool dependsOnStandardLibrary = false;
};

/// Prints the declarations of \p M to \p os in C++ language mode.
///
/// \returns Dependencies required by this module.
EmittedClangHeaderDependencyInfo printModuleContentsAsCxx(
    raw_ostream &os, ModuleDecl &M, CodiraToClangInteropContext &interopContext,
    bool requiresExposedAttribute, toolchain::StringSet<> &exposedModules);

} // end namespace language

#endif

