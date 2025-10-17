/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 7, 2022.
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
#include "api/units/data_size.h"

#include <cstdint>
#include <limits>

#include "test/gtest.h"

namespace webrtc {
namespace test {

TEST(DataSizeTest, ConstExpr) {
  constexpr int64_t kValue = 12345;
  constexpr DataSize kDataSizeZero = DataSize::Zero();
  constexpr DataSize kDataSizeInf = DataSize::Infinity();
  static_assert(kDataSizeZero.IsZero(), "");
  static_assert(kDataSizeInf.IsInfinite(), "");
  static_assert(kDataSizeInf.bytes_or(-1) == -1, "");
  static_assert(kDataSizeInf > kDataSizeZero, "");

  constexpr DataSize kDataSize = DataSize::Bytes(kValue);
  static_assert(kDataSize.bytes_or(-1) == kValue, "");

  EXPECT_EQ(kDataSize.bytes(), kValue);
}

TEST(DataSizeTest, GetBackSameValues) {
  const int64_t kValue = 123 * 8;
  EXPECT_EQ(DataSize::Bytes(kValue).bytes(), kValue);
}

TEST(DataSizeTest, IdentityChecks) {
  const int64_t kValue = 3000;
  EXPECT_TRUE(DataSize::Zero().IsZero());
  EXPECT_FALSE(DataSize::Bytes(kValue).IsZero());

  EXPECT_TRUE(DataSize::Infinity().IsInfinite());
  EXPECT_FALSE(DataSize::Zero().IsInfinite());
  EXPECT_FALSE(DataSize::Bytes(kValue).IsInfinite());

  EXPECT_FALSE(DataSize::Infinity().IsFinite());
  EXPECT_TRUE(DataSize::Bytes(kValue).IsFinite());
  EXPECT_TRUE(DataSize::Zero().IsFinite());
}

TEST(DataSizeTest, ComparisonOperators) {
  const int64_t kSmall = 450;
  const int64_t kLarge = 451;
  const DataSize small = DataSize::Bytes(kSmall);
  const DataSize large = DataSize::Bytes(kLarge);

  EXPECT_EQ(DataSize::Zero(), DataSize::Bytes(0));
  EXPECT_EQ(DataSize::Infinity(), DataSize::Infinity());
  EXPECT_EQ(small, small);
  EXPECT_LE(small, small);
  EXPECT_GE(small, small);
  EXPECT_NE(small, large);
  EXPECT_LE(small, large);
  EXPECT_LT(small, large);
  EXPECT_GE(large, small);
  EXPECT_GT(large, small);
  EXPECT_LT(DataSize::Zero(), small);
  EXPECT_GT(DataSize::Infinity(), large);
}

TEST(DataSizeTest, ConvertsToAndFromDouble) {
  const int64_t kValue = 128;
  const double kDoubleValue = static_cast<double>(kValue);

  EXPECT_EQ(DataSize::Bytes(kValue).bytes<double>(), kDoubleValue);
  EXPECT_EQ(DataSize::Bytes(kDoubleValue).bytes(), kValue);

  const double kInfinity = std::numeric_limits<double>::infinity();
  EXPECT_EQ(DataSize::Infinity().bytes<double>(), kInfinity);
  EXPECT_TRUE(DataSize::Bytes(kInfinity).IsInfinite());
}

TEST(DataSizeTest, MathOperations) {
  const int64_t kValueA = 450;
  const int64_t kValueB = 267;
  const DataSize size_a = DataSize::Bytes(kValueA);
  const DataSize size_b = DataSize::Bytes(kValueB);
  EXPECT_EQ((size_a + size_b).bytes(), kValueA + kValueB);
  EXPECT_EQ((size_a - size_b).bytes(), kValueA - kValueB);

  const int32_t kInt32Value = 123;
  const double kFloatValue = 123.0;
  EXPECT_EQ((size_a * kValueB).bytes(), kValueA * kValueB);
  EXPECT_EQ((size_a * kInt32Value).bytes(), kValueA * kInt32Value);
  EXPECT_EQ((size_a * kFloatValue).bytes(), kValueA * kFloatValue);

  EXPECT_EQ((size_a / 10).bytes(), kValueA / 10);
  EXPECT_EQ(size_a / size_b, static_cast<double>(kValueA) / kValueB);

  DataSize mutable_size = DataSize::Bytes(kValueA);
  mutable_size += size_b;
  EXPECT_EQ(mutable_size.bytes(), kValueA + kValueB);
  mutable_size -= size_a;
  EXPECT_EQ(mutable_size.bytes(), kValueB);
}
}  // namespace test
}  // namespace webrtc
