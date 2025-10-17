/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 22, 2025.
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

#include "language/IDE/CodeCompletion.h"
#include "gtest/gtest.h"

using namespace language;
using namespace ide;

static std::string replaceAtWithNull(const std::string &S) {
  std::string Result = S;
  for (char &C : Result) {
    if (C == '@')
      C = '\0';
  }
  return Result;
}

TEST(CodeCompletionToken, FindInEmptyFile) {
  std::string Source = "";
  unsigned Offset;
  std::string Clean = removeCodeCompletionTokens(Source, "A", &Offset);
  EXPECT_EQ(~0U, Offset);
  EXPECT_EQ("", Clean);
}

TEST(CodeCompletionToken, FindNonExistent) {
  std::string Source = "fn zzz() {}";
  unsigned Offset;
  std::string Clean = removeCodeCompletionTokens(Source, "A", &Offset);
  EXPECT_EQ(~0U, Offset);
  EXPECT_EQ(Source, Clean);
}

TEST(CodeCompletionToken, RemovesOtherTokens) {
  std::string Source = "fn zzz() {#^B^#}";
  unsigned Offset;
  std::string Clean = removeCodeCompletionTokens(Source, "A", &Offset);
  EXPECT_EQ(~0U, Offset);
  EXPECT_EQ("fn zzz() {}", Clean);
}

TEST(CodeCompletionToken, FindBegin) {
  std::string Source = "#^A^# fn";
  unsigned Offset;
  std::string Clean = removeCodeCompletionTokens(Source, "A", &Offset);
  EXPECT_EQ(0U, Offset);
  EXPECT_EQ(replaceAtWithNull("@ fn"), Clean);
}

TEST(CodeCompletionToken, FindEnd) {
  std::string Source = "fn #^A^#";
  unsigned Offset;
  std::string Clean = removeCodeCompletionTokens(Source, "A", &Offset);
  EXPECT_EQ(5U, Offset);
  EXPECT_EQ(replaceAtWithNull("fn @"), Clean);
}

TEST(CodeCompletionToken, FindSingleLine) {
  std::string Source = "fn zzz() {#^A^#}";
  unsigned Offset;
  std::string Clean = removeCodeCompletionTokens(Source, "A", &Offset);
  EXPECT_EQ(12U, Offset);
  EXPECT_EQ(replaceAtWithNull("fn zzz() {@}"), Clean);
}

TEST(CodeCompletionToken, FindMultiline) {
  std::string Source =
      "fn zzz() {\n"
      "  1 + #^A^#\r\n"
      "}\n";
  unsigned Offset;
  std::string Clean = removeCodeCompletionTokens(Source, "A", &Offset);
  EXPECT_EQ(19U, Offset);
  EXPECT_EQ(replaceAtWithNull("fn zzz() {\n  1 + @\r\n}\n"), Clean);
}

