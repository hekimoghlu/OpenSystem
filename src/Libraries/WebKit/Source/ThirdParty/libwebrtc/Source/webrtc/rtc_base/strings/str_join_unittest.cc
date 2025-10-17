/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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
#include "rtc_base/strings/str_join.h"

#include <string>
#include <utility>
#include <vector>

#include "test/gtest.h"

namespace webrtc {
namespace {

TEST(StrJoinTest, CanJoinStringsFromVector) {
  std::vector<std::string> strings = {"Hello", "World"};
  std::string s = StrJoin(strings, " ");
  EXPECT_EQ(s, "Hello World");
}

TEST(StrJoinTest, CanJoinNumbersFromArray) {
  std::array<int, 3> numbers = {1, 2, 3};
  std::string s = StrJoin(numbers, ",");
  EXPECT_EQ(s, "1,2,3");
}

TEST(StrJoinTest, CanFormatElementsWhileJoining) {
  std::vector<std::pair<std::string, std::string>> pairs = {
      {"hello", "world"}, {"foo", "bar"}, {"fum", "gazonk"}};
  std::string s = StrJoin(pairs, ",",
                          [&](rtc::StringBuilder& sb,
                              const std::pair<std::string, std::string>& p) {
                            sb << p.first << "=" << p.second;
                          });
  EXPECT_EQ(s, "hello=world,foo=bar,fum=gazonk");
}

}  // namespace
}  // namespace webrtc
