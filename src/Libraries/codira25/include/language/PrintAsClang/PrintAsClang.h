/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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

//===--- PrintAsClang.h - Emit a header file for a Codira AST ----*- C++ -*-===//
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

#ifndef LANGUAGE_PRINTASCLANG_H
#define LANGUAGE_PRINTASCLANG_H

#include "language/Basic/Toolchain.h"
#include "language/AST/AttrKind.h"
#include "language/AST/Identifier.h"

namespace language::Core {
class HeaderSearch;
}

namespace language {
class FrontendOptions;
class IRGenOptions;
class ModuleDecl;
class ValueDecl;

/// Print the exposed declarations in a module into a Clang header.
///
/// The Objective-C compatible declarations are printed into a block that
/// ensures that those declarations are only usable when the header is
/// compiled in Objective-C mode.
/// The C++ compatible declarations are printed into a block that ensures
/// that those declarations are only usable when the header is compiled in
/// C++ mode.
///
/// Returns true on error.
bool printAsClangHeader(raw_ostream &out, ModuleDecl *M,
                        StringRef bridgingHeader,
                        const FrontendOptions &frontendOpts,
                        const IRGenOptions &irGenOpts,
                        language::Core::HeaderSearch &headerSearchInfo);
}

#endif
