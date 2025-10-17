/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
#include "api/video/video_adaptation_counters.h"

#include "test/gtest.h"

namespace webrtc {

TEST(AdaptationCountersTest, Addition) {
  VideoAdaptationCounters a{0, 0};
  VideoAdaptationCounters b{1, 2};
  VideoAdaptationCounters total = a + b;
  EXPECT_EQ(1, total.resolution_adaptations);
  EXPECT_EQ(2, total.fps_adaptations);
}

TEST(AdaptationCountersTest, Equality) {
  VideoAdaptationCounters a{1, 2};
  VideoAdaptationCounters b{2, 1};
  EXPECT_EQ(a, a);
  EXPECT_NE(a, b);
}

}  // namespace webrtc
