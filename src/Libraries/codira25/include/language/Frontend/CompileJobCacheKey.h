/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 1, 2021.
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

//===--- CompileJobCacheKey.h - compile cache key methods -------*- C++ -*-===//
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
// This file contains declarations of utility methods for creating cache keys
// for compilation jobs.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_COMPILEJOBCACHEKEY_H
#define LANGUAGE_COMPILEJOBCACHEKEY_H

#include "language/AST/DiagnosticEngine.h"
#include "language/Basic/FileTypes.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/CAS/CASReference.h"
#include "toolchain/CAS/ObjectStore.h"
#include "toolchain/Support/Error.h"
#include "toolchain/Support/raw_ostream.h"

namespace language {

/// Compute CompileJobBaseKey from language-frontend command-line arguments.
/// CompileJobBaseKey represents the core inputs and arguments, and is used as a
/// base to compute keys for each compiler outputs.
// TODO: switch to create key from CompilerInvocation after we can canonicalize
// arguments.
toolchain::Expected<toolchain::cas::ObjectRef>
createCompileJobBaseCacheKey(toolchain::cas::ObjectStore &CAS,
                             ArrayRef<const char *> Args);

/// Compute CompileJobKey for the compiler outputs. The key for the output
/// is computed from the base key for the compilation and the input file index
/// which is the index for the input among all the input files (not just the
/// output producing inputs).
toolchain::Expected<toolchain::cas::ObjectRef>
createCompileJobCacheKeyForOutput(toolchain::cas::ObjectStore &CAS,
                                  toolchain::cas::ObjectRef BaseKey,
                                  unsigned InputIndex);

/// Print the CompileJobKey for debugging purpose.
toolchain::Error printCompileJobCacheKey(toolchain::cas::ObjectStore &CAS,
                                    toolchain::cas::ObjectRef Key,
                                    toolchain::raw_ostream &os);

} // namespace language

#endif
