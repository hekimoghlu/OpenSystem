/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 24, 2021.
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

//===--- ImmediateImpl.h - Support functions for immediate mode -*- C++ -*-===//
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

#ifndef LANGUAGE_IMMEDIATEIMPL_H
#define LANGUAGE_IMMEDIATEIMPL_H

#include "language/AST/LinkLibrary.h"
#include "language/AST/SearchPathOptions.h"
#include "language/Basic/Toolchain.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallPtrSet.h"
#include "toolchain/ADT/SmallVector.h"

namespace toolchain {
  class Function;
  class Module;
}

namespace language {
  class CompilerInstance;
  class DiagnosticEngine;
  class GeneratedModule;
  class IRGenOptions;
  class ModuleDecl;
  class SILOptions;

namespace immediate {

/// Returns a handle to the runtime suitable for other \c dlsym or \c dlclose
/// calls or \c null if an error occurred.
///
/// \param runtimeLibPaths Paths to search for stdlib dylibs.
void *loadCodiraRuntime(ArrayRef<std::string> runtimeLibPaths);
bool tryLoadLibraries(ArrayRef<LinkLibrary> LinkLibraries,
                      SearchPathOptions SearchPathOpts,
                      DiagnosticEngine &Diags);
bool linkLLVMModules(toolchain::Module *Module,
                     std::unique_ptr<toolchain::Module> &&SubModule);
bool autolinkImportedModules(ModuleDecl *M, const IRGenOptions &IRGenOpts);

} // end namespace immediate
} // end namespace language

#endif

