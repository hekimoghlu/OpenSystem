/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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

//===--- Markup.h - Markup --------------------------------------*- C++ -*-===//
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

#ifndef LANGUAGE_MARKUP_MARKUP_H
#define LANGUAGE_MARKUP_MARKUP_H

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/Support/Allocator.h"
#include "toolchain/Support/raw_ostream.h"
#include "language/Basic/SourceLoc.h"
#include "language/Markup/AST.h"
#include "language/Markup/LineList.h"

namespace language {

struct RawComment;

namespace markup {

class LineList;

class MarkupContext final {
  toolchain::BumpPtrAllocator Allocator;

public:
  void *allocate(unsigned long Bytes, unsigned Alignment) {
    return Allocator.Allocate(Bytes, Alignment);
  }

  template <typename T, typename It>
  T *allocateCopy(It Begin, It End) {
    T *Res =
    static_cast<T *>(allocate(sizeof(T) * (End - Begin), alignof(T)));
    for (unsigned i = 0; Begin != End; ++Begin, ++i)
      new (Res + i) T(*Begin);
    return Res;
  }

  template <typename T>
  MutableArrayRef<T> allocateCopy(ArrayRef<T> Array) {
    return MutableArrayRef<T>(allocateCopy<T>(Array.begin(), Array.end()),
                              Array.size());
  }

  StringRef allocateCopy(StringRef Str) {
    ArrayRef<char> Result =
        allocateCopy(toolchain::ArrayRef(Str.data(), Str.size()));
    return StringRef(Result.data(), Result.size());
  }

  LineList getLineList(language::RawComment RC);
};

Document *parseDocument(MarkupContext &MC, StringRef String);
Document *parseDocument(MarkupContext &MC, LineList &LL);

} // namespace markup
} // namespace language

#endif // LANGUAGE_MARKUP_MARKUP_H
