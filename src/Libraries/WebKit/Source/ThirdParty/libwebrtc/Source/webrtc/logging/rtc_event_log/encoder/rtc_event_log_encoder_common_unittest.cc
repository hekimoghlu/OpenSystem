/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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
#include "logging/rtc_event_log/encoder/rtc_event_log_encoder_common.h"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "test/gtest.h"

namespace webrtc_event_logging {
namespace {

template <typename T>
class SignednessConversionTest : public ::testing::Test {
 public:
  static_assert(std::is_integral<T>::value, "");
  static_assert(std::is_signed<T>::value, "");
};

TYPED_TEST_SUITE_P(SignednessConversionTest);

TYPED_TEST_P(SignednessConversionTest, CorrectlyConvertsLegalValues) {
  using T = TypeParam;
  std::vector<T> legal_values = {std::numeric_limits<T>::min(),
                                 std::numeric_limits<T>::min() + 1,
                                 -1,
                                 0,
                                 1,
                                 std::numeric_limits<T>::max() - 1,
                                 std::numeric_limits<T>::max()};
  for (T val : legal_values) {
    const auto unsigned_val = ToUnsigned(val);
    T signed_val;
    ASSERT_TRUE(ToSigned<T>(unsigned_val, &signed_val))
        << "Failed on " << static_cast<uint64_t>(unsigned_val) << ".";
    EXPECT_EQ(val, signed_val)
        << "Failed on " << static_cast<uint64_t>(unsigned_val) << ".";
  }
}

TYPED_TEST_P(SignednessConversionTest, FailsOnConvertingIllegalValues) {
  using T = TypeParam;

  // Note that a signed integer whose width is N bits, has N-1 digits.
  constexpr bool width_is_64 = std::numeric_limits<T>::digits == 63;

  if (width_is_64) {
    return;  // Test irrelevant; illegal values do not exist.
  }

  const uint64_t max_legal_value = ToUnsigned(static_cast<T>(-1));

  const std::vector<uint64_t> illegal_values = {
      max_legal_value + 1u, max_legal_value + 2u,
      std::numeric_limits<uint64_t>::max() - 1u,
      std::numeric_limits<uint64_t>::max()};

  for (uint64_t unsigned_val : illegal_values) {
    T signed_val;
    EXPECT_FALSE(ToSigned<T>(unsigned_val, &signed_val))
        << "Failed on " << static_cast<uint64_t>(unsigned_val) << ".";
  }
}

REGISTER_TYPED_TEST_SUITE_P(SignednessConversionTest,
                            CorrectlyConvertsLegalValues,
                            FailsOnConvertingIllegalValues);

using Types = ::testing::Types<int8_t, int16_t, int32_t, int64_t>;

INSTANTIATE_TYPED_TEST_SUITE_P(_, SignednessConversionTest, Types);

}  // namespace
}  // namespace webrtc_event_logging
