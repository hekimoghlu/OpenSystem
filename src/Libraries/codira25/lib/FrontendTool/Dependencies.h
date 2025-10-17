/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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

//===--- Dependencies.h -- Unified header for dependency tracing utilities --===//
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

#ifndef LANGUAGE_FRONTENDTOOL_DEPENDENCIES_H
#define LANGUAGE_FRONTENDTOOL_DEPENDENCIES_H

namespace toolchain::vfs {
class OutputBackend;
} // namespace toolchain::vfs

namespace language {

class DependencyTracker;
class FrontendOptions;
class InputFile;
class ModuleDecl;
class CompilerInstance;

/// Emit the names of the modules imported by \c mainModule.
bool emitImportedModules(ModuleDecl *mainModule, const FrontendOptions &opts,
                         toolchain::vfs::OutputBackend &backend);
bool emitLoadedModuleTraceIfNeeded(ModuleDecl *mainModule,
                                   DependencyTracker *depTracker,
                                   const FrontendOptions &opts,
                                   const InputFile &input);

bool emitFineModuleTraceIfNeeded(CompilerInstance &Instance,
                                 const FrontendOptions &opts);

} // end namespace language

#endif
