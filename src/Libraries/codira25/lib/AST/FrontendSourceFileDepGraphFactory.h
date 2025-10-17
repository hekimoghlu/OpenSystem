/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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

//===----- FrontendSourceFileDepGraphFactory.h ------------------*- C++ -*-===//
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

#ifndef FrontendSourceFileDepGraphFactory_h
#define FrontendSourceFileDepGraphFactory_h

#include "language/AST/AbstractSourceFileDepGraphFactory.h"
#include "toolchain/Support/VirtualOutputBackend.h"
namespace language {
namespace fine_grained_dependencies {

/// Constructs a SourceFileDepGraph from a *real* \c SourceFile
/// Reads the information provided by the frontend and builds the
/// SourceFileDepGraph

class FrontendSourceFileDepGraphFactory
    : public AbstractSourceFileDepGraphFactory {
  const SourceFile *SF;
  const DependencyTracker &depTracker;

public:
  FrontendSourceFileDepGraphFactory(const SourceFile *SF,
                                    toolchain::vfs::OutputBackend &backend,
                                    StringRef outputPath,
                                    const DependencyTracker &depTracker,
                                    bool alsoEmitDotFile);

  ~FrontendSourceFileDepGraphFactory() override = default;

private:
  void addAllDefinedDecls() override;
  void addAllUsedDecls() override;
};

class ModuleDepGraphFactory : public AbstractSourceFileDepGraphFactory {
  const ModuleDecl *Mod;

public:
  ModuleDepGraphFactory(toolchain::vfs::OutputBackend &backend,
                        const ModuleDecl *Mod, bool emitDot);

  ~ModuleDepGraphFactory() override = default;

private:
  void addAllDefinedDecls() override;
  void addAllUsedDecls() override {}
};

} // namespace fine_grained_dependencies
} // namespace language

#endif /* FrontendSourceFileDepGraphFactory_h */
