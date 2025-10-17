/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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
#include "modules/audio_processing/agc2/agc2_testing_common.h"

#include "rtc_base/gunit.h"

namespace webrtc {

TEST(GainController2TestingCommon, LinSpace) {
  std::vector<double> points1 = test::LinSpace(-1.0, 2.0, 4);
  const std::vector<double> expected_points1{{-1.0, 0.0, 1.0, 2.0}};
  EXPECT_EQ(expected_points1, points1);

  std::vector<double> points2 = test::LinSpace(0.0, 1.0, 4);
  const std::vector<double> expected_points2{{0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0}};
  EXPECT_EQ(points2, expected_points2);
}

}  // namespace webrtc
