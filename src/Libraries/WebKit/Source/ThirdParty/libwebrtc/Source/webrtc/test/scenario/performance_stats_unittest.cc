/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 22, 2023.
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
#include "test/scenario/performance_stats.h"

#include "test/gtest.h"

namespace webrtc {
namespace test {

TEST(EventRateCounter, ReturnsCorrectTotalDuration) {
  EventRateCounter event_rate_counter;
  EXPECT_EQ(event_rate_counter.TotalDuration(), TimeDelta::Zero());
  event_rate_counter.AddEvent(Timestamp::Seconds(1));
  EXPECT_EQ(event_rate_counter.TotalDuration(), TimeDelta::Zero());
  event_rate_counter.AddEvent(Timestamp::Seconds(2));
  EXPECT_EQ(event_rate_counter.TotalDuration(), TimeDelta::Seconds(1));
}

}  // namespace test
}  // namespace webrtc
