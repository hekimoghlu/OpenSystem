/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 3, 2022.
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
#include "rtc_base/string_utils.h"

#include "test/gtest.h"

namespace rtc {

TEST(string_toHexTest, ToHex) {
  EXPECT_EQ(ToHex(0), "0");
  EXPECT_EQ(ToHex(0X1243E), "1243e");
  EXPECT_EQ(ToHex(-20), "ffffffec");
}

#if defined(WEBRTC_WIN)

TEST(string_toutf, Empty) {
  char empty_string[] = "";
  EXPECT_TRUE(ToUtf16(empty_string, 0u).empty());
  wchar_t empty_wchar[] = L"";
  EXPECT_TRUE(ToUtf8(empty_wchar, 0u).empty());
}

#endif  // WEBRTC_WIN

TEST(CompileTimeString, MakeActsLikeAString) {
  EXPECT_STREQ(MakeCompileTimeString("abc123"), "abc123");
}

TEST(CompileTimeString, ConvertibleToStdString) {
  EXPECT_EQ(std::string(MakeCompileTimeString("abab")), "abab");
}

namespace detail {
constexpr bool StringEquals(const char* a, const char* b) {
  while (*a && *a == *b)
    a++, b++;
  return *a == *b;
}
}  // namespace detail

static_assert(detail::StringEquals(MakeCompileTimeString("handellm"),
                                   "handellm"),
              "String should initialize.");

static_assert(detail::StringEquals(MakeCompileTimeString("abc123").Concat(
                                       MakeCompileTimeString("def456ghi")),
                                   "abc123def456ghi"),
              "Strings should concatenate.");

}  // namespace rtc
