/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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
#include "../unit_test/unit_test.h"
#include "libyuv/basic_types.h"

namespace libyuv {

TEST_F(LibYUVBaseTest, SizeOfTypes) {
  int8_t i8 = -1;
  uint8_t u8 = 1u;
  int16_t i16 = -1;
  uint16_t u16 = 1u;
  int32_t i32 = -1;
  uint32_t u32 = 1u;
  int64_t i64 = -1;
  uint64_t u64 = 1u;
  EXPECT_EQ(1u, sizeof(i8));
  EXPECT_EQ(1u, sizeof(u8));
  EXPECT_EQ(2u, sizeof(i16));
  EXPECT_EQ(2u, sizeof(u16));
  EXPECT_EQ(4u, sizeof(i32));
  EXPECT_EQ(4u, sizeof(u32));
  EXPECT_EQ(8u, sizeof(i64));
  EXPECT_EQ(8u, sizeof(u64));
  EXPECT_GT(0, i8);
  EXPECT_LT(0u, u8);
  EXPECT_GT(0, i16);
  EXPECT_LT(0u, u16);
  EXPECT_GT(0, i32);
  EXPECT_LT(0u, u32);
  EXPECT_GT(0, i64);
  EXPECT_LT(0u, u64);
}

}  // namespace libyuv
