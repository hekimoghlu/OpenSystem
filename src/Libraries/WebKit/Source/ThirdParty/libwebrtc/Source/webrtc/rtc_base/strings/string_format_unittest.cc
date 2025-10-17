/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 31, 2023.
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
#include "rtc_base/strings/string_format.h"

#include <vector>

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"
#include "rtc_base/string_encode.h"
#include "test/gtest.h"

namespace rtc {

TEST(StringFormatTest, Empty) {
  EXPECT_EQ("", StringFormat("%s", ""));
}

TEST(StringFormatTest, Misc) {
  EXPECT_EQ("123hello w", StringFormat("%3d%2s %1c", 123, "hello", 'w'));
  EXPECT_EQ("3 = three", StringFormat("%d = %s", 1 + 2, "three"));
}

TEST(StringFormatTest, MaxSizeShouldWork) {
  const int kSrcLen = 512;
  char str[kSrcLen];
  std::fill_n(str, kSrcLen, 'A');
  str[kSrcLen - 1] = 0;
  EXPECT_EQ(str, StringFormat("%s", str));
}

// Test that formating a string using `absl::string_view` works as expected
// whe using `%.*s`.
TEST(StringFormatTest, FormatStringView) {
  const std::string main_string("This is a substring test.");
  std::vector<absl::string_view> string_views = rtc::split(main_string, ' ');
  ASSERT_EQ(string_views.size(), 5u);

  const absl::string_view& sv = string_views[3];
  std::string formatted =
      StringFormat("We have a %.*s.", static_cast<int>(sv.size()), sv.data());
  EXPECT_EQ(formatted.compare("We have a substring."), 0);
}

}  // namespace rtc
