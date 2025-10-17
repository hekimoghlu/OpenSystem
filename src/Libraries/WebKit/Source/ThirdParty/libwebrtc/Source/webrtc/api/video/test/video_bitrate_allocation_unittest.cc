/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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
#include "api/video/video_bitrate_allocation.h"

#include <optional>
#include <vector>

#include "test/gtest.h"

namespace webrtc {
TEST(VideoBitrateAllocation, SimulcastTargetBitrate) {
  VideoBitrateAllocation bitrate;
  bitrate.SetBitrate(0, 0, 10000);
  bitrate.SetBitrate(0, 1, 20000);
  bitrate.SetBitrate(1, 0, 40000);
  bitrate.SetBitrate(1, 1, 80000);

  VideoBitrateAllocation layer0_bitrate;
  layer0_bitrate.SetBitrate(0, 0, 10000);
  layer0_bitrate.SetBitrate(0, 1, 20000);

  VideoBitrateAllocation layer1_bitrate;
  layer1_bitrate.SetBitrate(0, 0, 40000);
  layer1_bitrate.SetBitrate(0, 1, 80000);

  std::vector<std::optional<VideoBitrateAllocation>> layer_allocations =
      bitrate.GetSimulcastAllocations();

  EXPECT_EQ(layer0_bitrate, layer_allocations[0]);
  EXPECT_EQ(layer1_bitrate, layer_allocations[1]);
}

TEST(VideoBitrateAllocation, SimulcastTargetBitrateWithInactiveStream) {
  // Create bitrate allocation with bitrate only for the first and third stream.
  VideoBitrateAllocation bitrate;
  bitrate.SetBitrate(0, 0, 10000);
  bitrate.SetBitrate(0, 1, 20000);
  bitrate.SetBitrate(2, 0, 40000);
  bitrate.SetBitrate(2, 1, 80000);

  VideoBitrateAllocation layer0_bitrate;
  layer0_bitrate.SetBitrate(0, 0, 10000);
  layer0_bitrate.SetBitrate(0, 1, 20000);

  VideoBitrateAllocation layer2_bitrate;
  layer2_bitrate.SetBitrate(0, 0, 40000);
  layer2_bitrate.SetBitrate(0, 1, 80000);

  std::vector<std::optional<VideoBitrateAllocation>> layer_allocations =
      bitrate.GetSimulcastAllocations();

  EXPECT_EQ(layer0_bitrate, layer_allocations[0]);
  EXPECT_FALSE(layer_allocations[1]);
  EXPECT_EQ(layer2_bitrate, layer_allocations[2]);
}
}  // namespace webrtc
