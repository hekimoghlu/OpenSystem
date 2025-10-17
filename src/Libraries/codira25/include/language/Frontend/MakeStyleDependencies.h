/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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

//===--- MakeStyleDependencies.h -- header for emitting dependency files --===//
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
//  This file defines interface for emitting Make Style Dependency Files.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_FRONTEND_MAKESTYLEDEPENDENCIES_H
#define LANGUAGE_FRONTEND_MAKESTYLEDEPENDENCIES_H

#include "toolchain/ADT/StringRef.h"

namespace toolchain {
class raw_ostream;
} // namespace toolchain

namespace language {

class CompilerInstance;
class DiagnosticEngine;
class FrontendOptions;
class InputFile;

/// Emit make style dependency file from compiler instance if needed.
bool emitMakeDependenciesIfNeeded(CompilerInstance &instance,
                                  const InputFile &input);

/// Emit make style dependency file from a serialized buffer.
bool emitMakeDependenciesFromSerializedBuffer(toolchain::StringRef buffer,
                                              toolchain::raw_ostream &os,
                                              const FrontendOptions &opts,
                                              const InputFile &input,
                                              DiagnosticEngine &diags);

} // end namespace language

#endif
