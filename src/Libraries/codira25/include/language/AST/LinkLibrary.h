/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 24, 2024.
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

//===--- LinkLibrary.h - A module-level linker dependency -------*- C++ -*-===//
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

#ifndef LANGUAGE_AST_LINKLIBRARY_H
#define LANGUAGE_AST_LINKLIBRARY_H

#include "language/Basic/Toolchain.h"
#include "toolchain/ADT/StringRef.h"
#include <string>

namespace language {

// Must be kept in sync with diag::error_immediate_mode_missing_library.
enum class LibraryKind {
  Library = 0,
  Framework
};

/// Represents a linker dependency for an imported module.
// FIXME: This is basically a slightly more generic version of Clang's
// Module::LinkLibrary.
class LinkLibrary {
private:
  std::string Name;
  unsigned Kind : 1;
  unsigned Static : 1;
  unsigned ForceLoad : 1;

public:
  LinkLibrary(StringRef N, LibraryKind K, bool Static, bool forceLoad = false)
      : Name(N), Kind(static_cast<unsigned>(K)), Static(Static),
        ForceLoad(forceLoad) {
    assert(getKind() == K && "not enough bits for the kind");
  }

  LibraryKind getKind() const { return static_cast<LibraryKind>(Kind); }
  StringRef getName() const { return Name; }
  bool isStaticLibrary() const { return Static; }
  bool shouldForceLoad() const { return ForceLoad; }
};

} // end namespace language

#endif
