/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 11, 2023.
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

//===--- FrontendTool.h - Frontend control ----------------------*- C++ -*-===//
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
//
// This file provides a high-level API for interacting with the basic
// frontend tool operation.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_FRONTENDTOOL_H
#define LANGUAGE_FRONTENDTOOL_H

#include "language/Basic/Toolchain.h"

namespace toolchain {
class Module;
}

namespace language {
class CompilerInvocation;
class CompilerInstance;
class SILModule;

/// A simple observer of frontend activity.
///
/// Don't let this interface block enhancements to the frontend pipeline.
class FrontendObserver {
public:
  FrontendObserver() = default;
  virtual ~FrontendObserver() = default;

  /// The frontend has parsed the command line.
  virtual void parsedArgs(CompilerInvocation &invocation);

  /// The frontend has configured the compiler instance.
  virtual void configuredCompiler(CompilerInstance &instance);

  /// The frontend has performed semantic analysis.
  virtual void performedSemanticAnalysis(CompilerInstance &instance);

  /// The frontend has performed basic SIL generation.
  /// SIL diagnostic passes have not yet been applied.
  virtual void performedSILGeneration(SILModule &module);

  /// The frontend has executed the SIL optimization and diagnostics pipelines.
  virtual void performedSILProcessing(SILModule &module);

  // TODO: maybe enhance this interface to hear about IRGen and LLVM
  // progress.
};

namespace frontend {
namespace utils {
StringRef escapeForMake(StringRef raw, toolchain::SmallVectorImpl<char> &buffer);
}
}

/// Perform all the operations of the frontend, exactly as if invoked
/// with -frontend.
///
/// \param args the arguments to use as the arguments to the frontend
/// \param argv0 the name used as the frontend executable
/// \param mainAddr an address from the main executable
///
/// \return the exit value of the frontend: 0 or 1 on success unless
///   the frontend executes in immediate mode, in which case this will be
///   the exit value of the script, assuming it exits normally
int performFrontend(ArrayRef<const char *> args,
                    const char *argv0,
                    void *mainAddr,
                    FrontendObserver *observer = nullptr);

bool performCompileStepsPostSema(CompilerInstance &Instance, int &ReturnValue,
                                 FrontendObserver *observer);

} // namespace language

#endif
