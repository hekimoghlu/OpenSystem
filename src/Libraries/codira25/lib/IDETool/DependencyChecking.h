/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 16, 2025.
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

//===--- DependencyChecking.h ---------------------------------------------===//
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

#include "language/Basic/Toolchain.h"
#include "toolchain/ADT/StringMap.h"
#include "toolchain/Support/Chrono.h"
#include "toolchain/Support/VirtualFileSystem.h"
#include <optional>

namespace language {
class CompilerInstance;

namespace ide {
/// Cache hash code of the dependencies into \p Map . If \p excludeBufferID is
/// specified, other source files are considered "dependencies", otherwise all
/// source files are considered "current"
void cacheDependencyHashIfNeeded(CompilerInstance &CI,
                                 std::optional<unsigned> excludeBufferID,
                                 toolchain::StringMap<toolchain::hash_code> &Map);

/// Check if any dependent files are modified since \p timestamp. If
/// \p excludeBufferID is specified, other source files are considered
/// "dependencies", otherwise all source files are considered "current".
/// \p Map should be the map populated by \c cacheDependencyHashIfNeeded at the
/// previous dependency checking.
bool areAnyDependentFilesInvalidated(
    CompilerInstance &CI, toolchain::vfs::FileSystem &FS,
    std::optional<unsigned> excludeBufferID, toolchain::sys::TimePoint<> timestamp,
    const toolchain::StringMap<toolchain::hash_code> &Map);

} // namespace ide
} // namespace language
