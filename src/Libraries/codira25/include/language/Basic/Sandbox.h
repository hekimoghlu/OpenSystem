/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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

//===--- Sandbox.h ----------------------------------------------*- C++ -*-===//
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

#ifndef LANGUAGE_BASIC_SANDBOX_H
#define LANGUAGE_BASIC_SANDBOX_H

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Allocator.h"

namespace language {
namespace Sandbox {

/// Applies a sandbox invocation to the given command line (if the platform
/// supports it), and returns the modified command line. On platforms that don't
/// support sandboxing, the command line is returned unmodified.
///
/// - Parameters:
///   - command: The command line to sandbox (including executable as first
///              argument)
///   - strictness: The basic strictness level of the standbox.
///   - writableDirectories: Paths under which writing should be allowed, even
///     if they would otherwise be read-only based on the strictness or paths in
///     `readOnlyDirectories`.
///   - readOnlyDirectories: Paths under which writing should be denied, even if
///     they would have otherwise been allowed by the rules implied by the
///     strictness level.
bool apply(toolchain::SmallVectorImpl<toolchain::StringRef> &command,
           toolchain::BumpPtrAllocator &Alloc);

} // namespace Sandbox
} // namespace language

#endif // LANGUAGE_BASIC_SANDBOX_H
