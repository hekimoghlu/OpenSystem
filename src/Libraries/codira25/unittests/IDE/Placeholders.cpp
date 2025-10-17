/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 25, 2022.
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

#include "language/IDE/Utils.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

using namespace language;
using namespace ide;

static std::string replaceFromString(const std::string &S,
                                     bool *HadPH = nullptr) {
  auto Buf = toolchain::MemoryBuffer::getMemBufferCopy(S, "");

  Buf = ide::replacePlaceholders(std::move(Buf), HadPH);
  return Buf->getBuffer().str();
}

TEST(Placeholders, Replace) {
  std::string Source = "aa <#one#> bb <# two #>";
  std::string Result = "aa $_0____ bb $_1______";
  std::string BufName = "<the buffer>";
  auto Buf = toolchain::MemoryBuffer::getMemBufferCopy(Source, BufName);
  struct PH {
    std::string Full;
    std::string Content;
    std::string Ident;
  };
  std::vector<PH> PHs;
  PHs.push_back(PH{"<#one#>", "one", "$_0____"});
  PHs.push_back(PH{"<# two #>", " two ", "$_1______"});

  unsigned Index = 0;
  Buf = ide::replacePlaceholders(std::move(Buf),
    [&](const PlaceholderOccurrence &Occur) {
      EXPECT_EQ(PHs[Index].Full, std::string(Occur.FullPlaceholder));
      EXPECT_EQ(PHs[Index].Content, std::string(Occur.PlaceholderContent));
      EXPECT_EQ(PHs[Index].Ident, std::string(Occur.IdentifierReplacement));
      ++Index;
    });

  EXPECT_EQ(BufName, std::string(Buf->getBufferIdentifier()));
  EXPECT_EQ(Result, std::string(Buf->getBuffer()));
}

TEST(Placeholders, ReplaceNoCallback) {
  std::string Source = "aa <#one#> bb <# two #>";
  std::string Result = "aa $_0____ bb $_1______";
  bool HadOne;
  std::string Out = replaceFromString(Source, &HadOne);
  EXPECT_EQ(true, HadOne);
  EXPECT_EQ(Result, Out);
}

TEST(Placeholders, NoPlaceholder1) {
  std::string Source = "aa";
  bool HadOne;
  std::string Out = replaceFromString(Source, &HadOne);
  EXPECT_EQ(false, HadOne);
  EXPECT_EQ(Source, Out);
}

TEST(Placeholders, NoPlaceholder2) {
  bool HadOne;
  replaceFromString("<#", &HadOne);
  EXPECT_EQ(false, HadOne);
  replaceFromString("<#a#", &HadOne);
  EXPECT_EQ(false, HadOne);
  replaceFromString("<#a\n#>", &HadOne);
  EXPECT_EQ(false, HadOne);
  replaceFromString("< #a#>", &HadOne);
  EXPECT_EQ(false, HadOne);
}

TEST(Placeholders, Nested) {
  std::string Source = "<#<#aa#>#>";
  std::string Result = "<#$_0___#>";
  std::string Out = replaceFromString(Source);
  EXPECT_EQ(Result, Out);
}

TEST(Placeholders, TooShort) {
  std::string Source;
  for (unsigned i = 0; i != 102; ++i) {
    Source += "<##>\n";
  }
  std::string Out = replaceFromString(Source);
  std::string Last(StringRef(Out).substr(Out.size()-15));
  EXPECT_EQ("$_99\n$___\n$___\n", Last);
}
