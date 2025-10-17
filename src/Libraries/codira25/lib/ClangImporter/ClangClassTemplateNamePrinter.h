/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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

//===--- ClangClassTemplateNamePrinter.h ------------------------*- C++ -*-===//
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

#ifndef LANGUAGE_CLANG_TEMPLATE_NAME_PRINTER_H
#define LANGUAGE_CLANG_TEMPLATE_NAME_PRINTER_H

#include "ImportName.h"
#include "language/AST/ASTContext.h"
#include "language/Core/AST/DeclTemplate.h"

namespace language {
namespace importer {

/// Returns a Codira representation of a C++ class template specialization name,
/// e.g. "vector<CWideChar, allocator<CWideChar>>".
///
/// This expands the entire tree of template instantiation names recursively.
/// While printing deep instantiation levels might not increase readability, it
/// is important to do because the C++ templated class names get mangled,
/// therefore they must be unique for different instantiations.
///
/// This function does not instantiate any templates and does not modify the AST
/// in any way.
std::string printClassTemplateSpecializationName(
    const language::Core::ClassTemplateSpecializationDecl *decl, ASTContext &languageCtx,
    NameImporter *nameImporter, ImportNameVersion version);

} // namespace importer
} // namespace language

#endif // LANGUAGE_CLANG_TEMPLATE_NAME_PRINTER_H
