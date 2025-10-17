/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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
#include "modules/desktop_capture/rgba_color.h"

#include <cstdint>
#include <vector>

#include "test/gtest.h"

namespace webrtc {

TEST(RgbaColorTest, ConvertFromAndToUInt32) {
  static const std::vector<uint32_t> cases{
      0,         1000,       2693,       3725,       4097,      12532,
      19902,     27002,      27723,      30944,      65535,     65536,
      231194,    255985,     322871,     883798,     9585200,   12410056,
      12641940,  30496970,   105735668,  110117847,  482769275, 542368468,
      798173396, 2678656711, 3231043200, UINT32_MAX,
  };

  for (uint32_t value : cases) {
    RgbaColor left(value);
    ASSERT_EQ(left.ToUInt32(), value);
    RgbaColor right(left);
    ASSERT_EQ(left.ToUInt32(), right.ToUInt32());
  }
}

TEST(RgbaColorTest, AlphaChannelEquality) {
  RgbaColor left(10, 10, 10, 0);
  RgbaColor right(10, 10, 10, 255);
  ASSERT_EQ(left, right);
  right.alpha = 128;
  ASSERT_NE(left, right);
}

}  // namespace webrtc
