/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 28, 2022.
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
#include "api/video/color_space.h"

#include <stdint.h>

#include "test/gtest.h"

namespace webrtc {
TEST(ColorSpace, TestSettingPrimariesFromUint8) {
  ColorSpace color_space;
  EXPECT_TRUE(color_space.set_primaries_from_uint8(
      static_cast<uint8_t>(ColorSpace::PrimaryID::kBT470BG)));
  EXPECT_EQ(ColorSpace::PrimaryID::kBT470BG, color_space.primaries());
  EXPECT_FALSE(color_space.set_primaries_from_uint8(3));
  EXPECT_FALSE(color_space.set_primaries_from_uint8(23));
  EXPECT_FALSE(color_space.set_primaries_from_uint8(64));
}

TEST(ColorSpace, TestSettingTransferFromUint8) {
  ColorSpace color_space;
  EXPECT_TRUE(color_space.set_transfer_from_uint8(
      static_cast<uint8_t>(ColorSpace::TransferID::kBT2020_10)));
  EXPECT_EQ(ColorSpace::TransferID::kBT2020_10, color_space.transfer());
  EXPECT_FALSE(color_space.set_transfer_from_uint8(3));
  EXPECT_FALSE(color_space.set_transfer_from_uint8(19));
  EXPECT_FALSE(color_space.set_transfer_from_uint8(128));
}

TEST(ColorSpace, TestSettingMatrixFromUint8) {
  ColorSpace color_space;
  EXPECT_TRUE(color_space.set_matrix_from_uint8(
      static_cast<uint8_t>(ColorSpace::MatrixID::kCDNCLS)));
  EXPECT_EQ(ColorSpace::MatrixID::kCDNCLS, color_space.matrix());
  EXPECT_FALSE(color_space.set_matrix_from_uint8(3));
  EXPECT_FALSE(color_space.set_matrix_from_uint8(15));
  EXPECT_FALSE(color_space.set_matrix_from_uint8(255));
}

TEST(ColorSpace, TestSettingRangeFromUint8) {
  ColorSpace color_space;
  EXPECT_TRUE(color_space.set_range_from_uint8(
      static_cast<uint8_t>(ColorSpace::RangeID::kFull)));
  EXPECT_EQ(ColorSpace::RangeID::kFull, color_space.range());
  EXPECT_FALSE(color_space.set_range_from_uint8(4));
}

TEST(ColorSpace, TestSettingChromaSitingHorizontalFromUint8) {
  ColorSpace color_space;
  EXPECT_TRUE(color_space.set_chroma_siting_horizontal_from_uint8(
      static_cast<uint8_t>(ColorSpace::ChromaSiting::kCollocated)));
  EXPECT_EQ(ColorSpace::ChromaSiting::kCollocated,
            color_space.chroma_siting_horizontal());
  EXPECT_FALSE(color_space.set_chroma_siting_horizontal_from_uint8(3));
}

TEST(ColorSpace, TestSettingChromaSitingVerticalFromUint8) {
  ColorSpace color_space;
  EXPECT_TRUE(color_space.set_chroma_siting_vertical_from_uint8(
      static_cast<uint8_t>(ColorSpace::ChromaSiting::kHalf)));
  EXPECT_EQ(ColorSpace::ChromaSiting::kHalf,
            color_space.chroma_siting_vertical());
  EXPECT_FALSE(color_space.set_chroma_siting_vertical_from_uint8(3));
}

TEST(ColorSpace, TestAsStringFunction) {
  ColorSpace color_space(
      ColorSpace::PrimaryID::kBT709, ColorSpace::TransferID::kBT709,
      ColorSpace::MatrixID::kBT709, ColorSpace::RangeID::kLimited);
  EXPECT_EQ(
      color_space.AsString(),
      "{primaries:kBT709, transfer:kBT709, matrix:kBT709, range:kLimited}");
}

}  // namespace webrtc
