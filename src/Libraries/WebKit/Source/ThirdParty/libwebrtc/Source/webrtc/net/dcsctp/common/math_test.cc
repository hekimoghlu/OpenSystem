/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 24, 2022.
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
#include "net/dcsctp/common/math.h"

#include "test/gmock.h"

namespace dcsctp {
namespace {

TEST(MathUtilTest, CanRoundUpTo4) {
  // Signed numbers
  EXPECT_EQ(RoundUpTo4(static_cast<int>(-5)), -4);
  EXPECT_EQ(RoundUpTo4(static_cast<int>(-4)), -4);
  EXPECT_EQ(RoundUpTo4(static_cast<int>(-3)), 0);
  EXPECT_EQ(RoundUpTo4(static_cast<int>(-2)), 0);
  EXPECT_EQ(RoundUpTo4(static_cast<int>(-1)), 0);
  EXPECT_EQ(RoundUpTo4(static_cast<int>(0)), 0);
  EXPECT_EQ(RoundUpTo4(static_cast<int>(1)), 4);
  EXPECT_EQ(RoundUpTo4(static_cast<int>(2)), 4);
  EXPECT_EQ(RoundUpTo4(static_cast<int>(3)), 4);
  EXPECT_EQ(RoundUpTo4(static_cast<int>(4)), 4);
  EXPECT_EQ(RoundUpTo4(static_cast<int>(5)), 8);
  EXPECT_EQ(RoundUpTo4(static_cast<int>(6)), 8);
  EXPECT_EQ(RoundUpTo4(static_cast<int>(7)), 8);
  EXPECT_EQ(RoundUpTo4(static_cast<int>(8)), 8);
  EXPECT_EQ(RoundUpTo4(static_cast<int64_t>(10000000000)), 10000000000);
  EXPECT_EQ(RoundUpTo4(static_cast<int64_t>(10000000001)), 10000000004);

  // Unsigned numbers
  EXPECT_EQ(RoundUpTo4(static_cast<unsigned int>(0)), 0u);
  EXPECT_EQ(RoundUpTo4(static_cast<unsigned int>(1)), 4u);
  EXPECT_EQ(RoundUpTo4(static_cast<unsigned int>(2)), 4u);
  EXPECT_EQ(RoundUpTo4(static_cast<unsigned int>(3)), 4u);
  EXPECT_EQ(RoundUpTo4(static_cast<unsigned int>(4)), 4u);
  EXPECT_EQ(RoundUpTo4(static_cast<unsigned int>(5)), 8u);
  EXPECT_EQ(RoundUpTo4(static_cast<unsigned int>(6)), 8u);
  EXPECT_EQ(RoundUpTo4(static_cast<unsigned int>(7)), 8u);
  EXPECT_EQ(RoundUpTo4(static_cast<unsigned int>(8)), 8u);
  EXPECT_EQ(RoundUpTo4(static_cast<uint64_t>(10000000000)), 10000000000u);
  EXPECT_EQ(RoundUpTo4(static_cast<uint64_t>(10000000001)), 10000000004u);
}

TEST(MathUtilTest, CanRoundDownTo4) {
  // Signed numbers
  EXPECT_EQ(RoundDownTo4(static_cast<int>(-5)), -8);
  EXPECT_EQ(RoundDownTo4(static_cast<int>(-4)), -4);
  EXPECT_EQ(RoundDownTo4(static_cast<int>(-3)), -4);
  EXPECT_EQ(RoundDownTo4(static_cast<int>(-2)), -4);
  EXPECT_EQ(RoundDownTo4(static_cast<int>(-1)), -4);
  EXPECT_EQ(RoundDownTo4(static_cast<int>(0)), 0);
  EXPECT_EQ(RoundDownTo4(static_cast<int>(1)), 0);
  EXPECT_EQ(RoundDownTo4(static_cast<int>(2)), 0);
  EXPECT_EQ(RoundDownTo4(static_cast<int>(3)), 0);
  EXPECT_EQ(RoundDownTo4(static_cast<int>(4)), 4);
  EXPECT_EQ(RoundDownTo4(static_cast<int>(5)), 4);
  EXPECT_EQ(RoundDownTo4(static_cast<int>(6)), 4);
  EXPECT_EQ(RoundDownTo4(static_cast<int>(7)), 4);
  EXPECT_EQ(RoundDownTo4(static_cast<int>(8)), 8);
  EXPECT_EQ(RoundDownTo4(static_cast<int64_t>(10000000000)), 10000000000);
  EXPECT_EQ(RoundDownTo4(static_cast<int64_t>(10000000001)), 10000000000);

  // Unsigned numbers
  EXPECT_EQ(RoundDownTo4(static_cast<unsigned int>(0)), 0u);
  EXPECT_EQ(RoundDownTo4(static_cast<unsigned int>(1)), 0u);
  EXPECT_EQ(RoundDownTo4(static_cast<unsigned int>(2)), 0u);
  EXPECT_EQ(RoundDownTo4(static_cast<unsigned int>(3)), 0u);
  EXPECT_EQ(RoundDownTo4(static_cast<unsigned int>(4)), 4u);
  EXPECT_EQ(RoundDownTo4(static_cast<unsigned int>(5)), 4u);
  EXPECT_EQ(RoundDownTo4(static_cast<unsigned int>(6)), 4u);
  EXPECT_EQ(RoundDownTo4(static_cast<unsigned int>(7)), 4u);
  EXPECT_EQ(RoundDownTo4(static_cast<unsigned int>(8)), 8u);
  EXPECT_EQ(RoundDownTo4(static_cast<uint64_t>(10000000000)), 10000000000u);
  EXPECT_EQ(RoundDownTo4(static_cast<uint64_t>(10000000001)), 10000000000u);
}

TEST(MathUtilTest, IsDivisibleBy4) {
  // Signed numbers
  EXPECT_EQ(IsDivisibleBy4(static_cast<int>(-4)), true);
  EXPECT_EQ(IsDivisibleBy4(static_cast<int>(-3)), false);
  EXPECT_EQ(IsDivisibleBy4(static_cast<int>(-2)), false);
  EXPECT_EQ(IsDivisibleBy4(static_cast<int>(-1)), false);
  EXPECT_EQ(IsDivisibleBy4(static_cast<int>(0)), true);
  EXPECT_EQ(IsDivisibleBy4(static_cast<int>(1)), false);
  EXPECT_EQ(IsDivisibleBy4(static_cast<int>(2)), false);
  EXPECT_EQ(IsDivisibleBy4(static_cast<int>(3)), false);
  EXPECT_EQ(IsDivisibleBy4(static_cast<int>(4)), true);
  EXPECT_EQ(IsDivisibleBy4(static_cast<int>(5)), false);
  EXPECT_EQ(IsDivisibleBy4(static_cast<int>(6)), false);
  EXPECT_EQ(IsDivisibleBy4(static_cast<int>(7)), false);
  EXPECT_EQ(IsDivisibleBy4(static_cast<int>(8)), true);
  EXPECT_EQ(IsDivisibleBy4(static_cast<int64_t>(10000000000)), true);
  EXPECT_EQ(IsDivisibleBy4(static_cast<int64_t>(10000000001)), false);

  // Unsigned numbers
  EXPECT_EQ(IsDivisibleBy4(static_cast<unsigned int>(0)), true);
  EXPECT_EQ(IsDivisibleBy4(static_cast<unsigned int>(1)), false);
  EXPECT_EQ(IsDivisibleBy4(static_cast<unsigned int>(2)), false);
  EXPECT_EQ(IsDivisibleBy4(static_cast<unsigned int>(3)), false);
  EXPECT_EQ(IsDivisibleBy4(static_cast<unsigned int>(4)), true);
  EXPECT_EQ(IsDivisibleBy4(static_cast<unsigned int>(5)), false);
  EXPECT_EQ(IsDivisibleBy4(static_cast<unsigned int>(6)), false);
  EXPECT_EQ(IsDivisibleBy4(static_cast<unsigned int>(7)), false);
  EXPECT_EQ(IsDivisibleBy4(static_cast<unsigned int>(8)), true);
  EXPECT_EQ(IsDivisibleBy4(static_cast<uint64_t>(10000000000)), true);
  EXPECT_EQ(IsDivisibleBy4(static_cast<uint64_t>(10000000001)), false);
}

}  // namespace
}  // namespace dcsctp
