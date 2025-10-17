/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 21, 2024.
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
#include "rtc_base/string_to_number.h"

#include <stdint.h>

#include <limits>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "test/gtest.h"

namespace rtc {

namespace {
// clang-format off
using IntegerTypes =
    ::testing::Types<char,
                     signed char, unsigned char,       // NOLINT(runtime/int)
                     short,       unsigned short,      // NOLINT(runtime/int)
                     int,         unsigned int,        // NOLINT(runtime/int)
                     long,        unsigned long,       // NOLINT(runtime/int)
                     long long,   unsigned long long,  // NOLINT(runtime/int)
                     int8_t,      uint8_t,
                     int16_t,     uint16_t,
                     int32_t,     uint32_t,
                     int64_t,     uint64_t>;
// clang-format on

template <typename T>
class BasicNumberTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(BasicNumberTest);

TYPED_TEST_P(BasicNumberTest, TestValidNumbers) {
  using T = TypeParam;
  constexpr T min_value = std::numeric_limits<T>::lowest();
  constexpr T max_value = std::numeric_limits<T>::max();
  constexpr T zero_value = 0;
  const std::string min_string = std::to_string(min_value);
  const std::string max_string = std::to_string(max_value);
  EXPECT_EQ(min_value, StringToNumber<T>(min_string));
  EXPECT_EQ(min_value, StringToNumber<T>(min_string.c_str()));
  EXPECT_EQ(max_value, StringToNumber<T>(max_string));
  EXPECT_EQ(max_value, StringToNumber<T>(max_string.c_str()));
  EXPECT_EQ(zero_value, StringToNumber<T>("0"));
  EXPECT_EQ(zero_value, StringToNumber<T>("-0"));
  EXPECT_EQ(zero_value, StringToNumber<T>(std::string("-0000000000000")));
}

TYPED_TEST_P(BasicNumberTest, TestInvalidNumbers) {
  using T = TypeParam;
  // Value ranges aren't strictly enforced in this test, since that would either
  // require doctoring specific strings for each data type, which is a hassle
  // across platforms, or to be able to do addition of values larger than the
  // largest type, which is another hassle.
  constexpr T min_value = std::numeric_limits<T>::lowest();
  constexpr T max_value = std::numeric_limits<T>::max();
  // If the type supports negative values, make the large negative value
  // approximately ten times larger. If the type is unsigned, just use -2.
  const std::string too_low_string =
      (min_value == 0) ? "-2" : (std::to_string(min_value) + "1");
  // Make the large value approximately ten times larger than the maximum.
  const std::string too_large_string = std::to_string(max_value) + "1";
  EXPECT_EQ(std::nullopt, StringToNumber<T>(too_low_string));
  EXPECT_EQ(std::nullopt, StringToNumber<T>(too_low_string.c_str()));
  EXPECT_EQ(std::nullopt, StringToNumber<T>(too_large_string));
  EXPECT_EQ(std::nullopt, StringToNumber<T>(too_large_string.c_str()));
}

TYPED_TEST_P(BasicNumberTest, TestInvalidInputs) {
  using T = TypeParam;
  const char kInvalidCharArray[] = "Invalid string containing 47";
  const char kPlusMinusCharArray[] = "+-100";
  const char kNumberFollowedByCruft[] = "640x480";
  const char kEmbeddedNul[] = {'1', '2', '\0', '3', '4'};
  const char kBeginningEmbeddedNul[] = {'\0', '1', '2', '3', '4'};
  const char kTrailingEmbeddedNul[] = {'1', '2', '3', '4', '\0'};

  EXPECT_EQ(std::nullopt, StringToNumber<T>(kInvalidCharArray));
  EXPECT_EQ(std::nullopt, StringToNumber<T>(std::string(kInvalidCharArray)));
  EXPECT_EQ(std::nullopt, StringToNumber<T>(kPlusMinusCharArray));
  EXPECT_EQ(std::nullopt, StringToNumber<T>(std::string(kPlusMinusCharArray)));
  EXPECT_EQ(std::nullopt, StringToNumber<T>(kNumberFollowedByCruft));
  EXPECT_EQ(std::nullopt,
            StringToNumber<T>(std::string(kNumberFollowedByCruft)));
  EXPECT_EQ(std::nullopt, StringToNumber<T>(" 5"));
  EXPECT_EQ(std::nullopt, StringToNumber<T>(" - 5"));
  EXPECT_EQ(std::nullopt, StringToNumber<T>("- 5"));
  EXPECT_EQ(std::nullopt, StringToNumber<T>(" -5"));
  EXPECT_EQ(std::nullopt, StringToNumber<T>("5 "));
  // Test various types of empty inputs
  EXPECT_EQ(std::nullopt, StringToNumber<T>({nullptr, 0}));
  EXPECT_EQ(std::nullopt, StringToNumber<T>(""));
  EXPECT_EQ(std::nullopt, StringToNumber<T>(std::string()));
  EXPECT_EQ(std::nullopt, StringToNumber<T>(std::string("")));
  EXPECT_EQ(std::nullopt, StringToNumber<T>(absl::string_view()));
  EXPECT_EQ(std::nullopt, StringToNumber<T>(absl::string_view(nullptr, 0)));
  EXPECT_EQ(std::nullopt, StringToNumber<T>(absl::string_view("")));
  // Test strings with embedded nuls.
  EXPECT_EQ(std::nullopt, StringToNumber<T>(absl::string_view(
                              kEmbeddedNul, sizeof(kEmbeddedNul))));
  EXPECT_EQ(std::nullopt,
            StringToNumber<T>(absl::string_view(
                kBeginningEmbeddedNul, sizeof(kBeginningEmbeddedNul))));
  EXPECT_EQ(std::nullopt,
            StringToNumber<T>(absl::string_view(kTrailingEmbeddedNul,
                                                sizeof(kTrailingEmbeddedNul))));
}

REGISTER_TYPED_TEST_SUITE_P(BasicNumberTest,
                            TestValidNumbers,
                            TestInvalidNumbers,
                            TestInvalidInputs);

}  // namespace

INSTANTIATE_TYPED_TEST_SUITE_P(StringToNumberTest_Integers,
                               BasicNumberTest,
                               IntegerTypes);

TEST(StringToNumberTest, TestSpecificValues) {
  EXPECT_EQ(std::nullopt, StringToNumber<uint8_t>("256"));
  EXPECT_EQ(std::nullopt, StringToNumber<uint8_t>("-256"));
  EXPECT_EQ(std::nullopt, StringToNumber<int8_t>("256"));
  EXPECT_EQ(std::nullopt, StringToNumber<int8_t>("-256"));
}

}  // namespace rtc
