/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 1, 2021.
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

//===--- EditorPlaceholderTest.cpp ----------------------------------------===//
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

#include "language/Basic/EditorPlaceholder.h"
#include "gtest/gtest.h"
#include <optional>

using namespace language;

TEST(EditorPlaceholder, EditorPlaceholders) {
  const char *Text = "<#this is simple#>";
  EditorPlaceholderData Data = *parseEditorPlaceholder(Text);
  EXPECT_EQ(EditorPlaceholderKind::Basic, Data.Kind);
  EXPECT_EQ("this is simple", Data.Display);
  EXPECT_TRUE(Data.Type.empty());
  EXPECT_TRUE(Data.TypeForExpansion.empty());

  Text = "<#T##x: Int##Int#>";
  Data = *parseEditorPlaceholder(Text);
  EXPECT_EQ(EditorPlaceholderKind::Typed, Data.Kind);
  EXPECT_EQ("x: Int", Data.Display);
  EXPECT_EQ("Int", Data.Type);
  EXPECT_EQ("Int", Data.TypeForExpansion);

  Text = "<#T##x: Int##Blah##()->Int#>";
  Data = *parseEditorPlaceholder(Text);
  EXPECT_EQ(EditorPlaceholderKind::Typed, Data.Kind);
  EXPECT_EQ("x: Int", Data.Display);
  EXPECT_EQ("Blah", Data.Type);
  EXPECT_EQ("()->Int", Data.TypeForExpansion);

  Text = "<#T##Int#>";
  Data = *parseEditorPlaceholder(Text);
  EXPECT_EQ(EditorPlaceholderKind::Typed, Data.Kind);
  EXPECT_EQ("Int", Data.Display);
  EXPECT_EQ("Int", Data.Type);
  EXPECT_EQ("Int", Data.TypeForExpansion);
}

TEST(EditorPlaceholder, InvalidEditorPlaceholders) {
  const char *Text = "<#foo";
  std::optional<EditorPlaceholderData> DataOpt = parseEditorPlaceholder(Text);
  EXPECT_FALSE(DataOpt.has_value());

  Text = "foo#>";
  DataOpt = parseEditorPlaceholder(Text);
  EXPECT_FALSE(DataOpt.has_value());

  Text = "#foo#>";
  DataOpt = parseEditorPlaceholder(Text);
  EXPECT_FALSE(DataOpt.has_value());

  Text = " <#foo#>";
  DataOpt = parseEditorPlaceholder(Text);
  EXPECT_FALSE(DataOpt.has_value());
}
