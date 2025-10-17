/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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

//===--- SymbolGraphGen.h - Codira SymbolGraph Generator -------------------===//
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

#ifndef LANGUAGE_SYMBOLGRAPHGEN_SYMBOLGRAPHGEN_H
#define LANGUAGE_SYMBOLGRAPHGEN_SYMBOLGRAPHGEN_H

#include "language/AST/Module.h"
#include "language/AST/Type.h"
#include "SymbolGraphOptions.h"
#include "PathComponent.h"
#include "FragmentInfo.h"

namespace language {
class ValueDecl;

namespace symbolgraphgen {

/// Emit a Symbol Graph JSON file for a module.
int emitSymbolGraphForModule(ModuleDecl *M, const SymbolGraphOptions &Options);

/// Print a Symbol Graph containing a single node for the given decl to \p OS.
/// The \p ParentContexts out parameter will also be populated with information
/// about each parent context of the given decl, from outermost to innermost.
///
/// \returns \c EXIT_SUCCESS if the kind of the provided node is supported or
/// \c EXIT_FAILURE otherwise.
int printSymbolGraphForDecl(const ValueDecl *D, Type BaseTy,
                            bool InSynthesizedExtension,
                            const SymbolGraphOptions &Options,
                            toolchain::raw_ostream &OS,
                            SmallVectorImpl<PathComponent> &ParentContexts,
                            SmallVectorImpl<FragmentInfo> &FragmentInfo);

} // end namespace symbolgraphgen
} // end namespace language

#endif // LANGUAGE_SYMBOLGRAPHGEN_SYMBOLGRAPHGEN_H
