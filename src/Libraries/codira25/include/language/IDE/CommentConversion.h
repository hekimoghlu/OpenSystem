/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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

//===--- CommentConversion.h - Conversion of comments to other formats ----===//
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

#ifndef LANGUAGE_IDE_COMMENT_CONVERSION_H
#define LANGUAGE_IDE_COMMENT_CONVERSION_H

#include "language/Basic/Toolchain.h"
#include "language/AST/TypeOrExtensionDecl.h"
#include <memory>
#include <string>

namespace language {
class Decl;
class DocComment;

namespace ide {

/// If the declaration has a documentation comment, prints the comment to \p OS
/// in Clang-like XML format.
///
/// \returns true if the declaration has a documentation comment.
bool getDocumentationCommentAsXML(
  const Decl *D, raw_ostream &OS,
  TypeOrExtensionDecl SynthesizedTarget = TypeOrExtensionDecl());

/// If the declaration has a documentation comment, prints the comment to \p OS
/// in the form it's written in source.
///
/// \returns true if the declaration has a documentation comment.
bool getRawDocumentationComment(const Decl *D, raw_ostream &OS);

/// If the declaration has a documentation comment and a localization key,
/// print it into the given output stream and return true. Else, return false.
bool getLocalizationKey(const Decl *D, raw_ostream &OS);

/// Converts the given comment to Doxygen.
void getDocumentationCommentAsDoxygen(const DocComment *DC, raw_ostream &OS);

/// Extract and normalize text from the given comment.
std::string extractPlainTextFromComment(const StringRef Text);

/// Given the raw text in markup format, convert its content to xml.
bool convertMarkupToXML(StringRef Text, raw_ostream &OS);

} // namespace ide
} // namespace language

#endif // LANGUAGE_IDE_COMMENT_CONVERSION_H

