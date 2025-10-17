/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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

//===----------------------------------------------------------------------===//
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

#include "SourceKit/Support/ImmutableTextBuffer.h"
#include "gtest/gtest.h"

using namespace SourceKit;
using namespace toolchain;

TEST(EditableTextBuffer, Updates) {
  const char *Text = "hello world";
  size_t Length = strlen(Text);

  EditableTextBufferManager BufMgr;
  EditableTextBufferRef EdBuf = BufMgr.getOrCreateBuffer("/a/test", Text);
  ImmutableTextBufferRef Buf = EdBuf->getBuffer();

  EXPECT_EQ(Buf->getText(), Text);
  EXPECT_EQ(EdBuf->getSize(), Length);

  Buf = EdBuf->insert(6, "all ")->getBuffer();
  EXPECT_EQ(Buf->getText(), "hello all world");
  EXPECT_EQ(EdBuf->getSize(), strlen("hello all world"));

  Buf = EdBuf->erase(9, 6)->getBuffer();
  EXPECT_EQ(Buf->getText(), "hello all");
  EXPECT_EQ(EdBuf->getSize(), strlen("hello all"));

  Buf = EdBuf->replace(0, 5, "yo")->getBuffer();
  EXPECT_EQ(Buf->getText(), "yo all");
  EXPECT_EQ(EdBuf->getSize(), strlen("yo all"));

  EdBuf = BufMgr.resetBuffer("/a/test", Text);
  EdBuf->insert(6, "all ");
  EdBuf->erase(9, 6);
  EdBuf->replace(0, 5, "yo");
  EXPECT_EQ(EdBuf->getSize(), strlen("yo all"));
  Buf = EdBuf->getSnapshot()->getBuffer();
  EXPECT_EQ(Buf->getText(), "yo all");

  EXPECT_EQ(Buf->getFilename(), "/a/test");
}
