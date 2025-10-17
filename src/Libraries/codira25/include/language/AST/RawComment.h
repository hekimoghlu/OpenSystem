/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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

//===--- RawComment.h - Extraction of raw comments --------------*- C++ -*-===//
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

#ifndef LANGUAGE_AST_RAW_COMMENT_H
#define LANGUAGE_AST_RAW_COMMENT_H

#include "language/Basic/SourceLoc.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/StringRef.h"

namespace language {

class SourceFile;
class SourceManager;

struct SingleRawComment {
  enum class CommentKind {
    OrdinaryLine,  ///< Any normal // comments
    OrdinaryBlock, ///< Any normal /* */ comment
    LineDoc,       ///< \code /// stuff \endcode
    BlockDoc,      ///< \code /** stuff */ \endcode
  };

  CharSourceRange Range;
  StringRef RawText;

  unsigned Kind : 8;
  unsigned ColumnIndent : 16;

  SingleRawComment(CharSourceRange Range, const SourceManager &SourceMgr);
  SingleRawComment(StringRef RawText, unsigned ColumnIndent);

  SingleRawComment(const SingleRawComment &) = default;
  SingleRawComment &operator=(const SingleRawComment &) = default;

  CommentKind getKind() const TOOLCHAIN_READONLY {
    return static_cast<CommentKind>(Kind);
  }

  bool isOrdinary() const TOOLCHAIN_READONLY {
    return getKind() == CommentKind::OrdinaryLine ||
           getKind() == CommentKind::OrdinaryBlock;
  }

  bool isLine() const TOOLCHAIN_READONLY {
    return getKind() == CommentKind::OrdinaryLine ||
           getKind() == CommentKind::LineDoc;
  }

  bool isGyb() const {
    return RawText.starts_with("// ###");
  }
};

struct RawComment {
  ArrayRef<SingleRawComment> Comments;

  RawComment() {}
  RawComment(ArrayRef<SingleRawComment> Comments) : Comments(Comments) {}

  RawComment(const RawComment &) = default;
  RawComment &operator=(const RawComment &) = default;

  bool isEmpty() const {
    return Comments.empty();
  }

  CharSourceRange getCharSourceRange();
};

struct CommentInfo {
  StringRef Brief;
  RawComment Raw;
  uint32_t Group;
  uint32_t SourceOrder;
};

} // namespace language

#endif // TOOLCHAIN_LANGUAGE_AST_RAW_COMMENT_H

