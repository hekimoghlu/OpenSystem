/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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
#include "modules/rtp_rtcp/source/rtcp_packet/remote_estimate.h"

#include "test/gtest.h"

namespace webrtc {
namespace rtcp {
TEST(RemoteEstimateTest, EncodesCapacityBounds) {
  NetworkStateEstimate src;
  src.link_capacity_lower = DataRate::KilobitsPerSec(10);
  src.link_capacity_upper = DataRate::KilobitsPerSec(1000000);
  rtc::Buffer data = GetRemoteEstimateSerializer()->Serialize(src);
  NetworkStateEstimate dst;
  EXPECT_TRUE(GetRemoteEstimateSerializer()->Parse(data, &dst));
  EXPECT_EQ(src.link_capacity_lower, dst.link_capacity_lower);
  EXPECT_EQ(src.link_capacity_upper, dst.link_capacity_upper);
}

TEST(RemoteEstimateTest, ExpandsToPlusInfinity) {
  NetworkStateEstimate src;
  // White box testing: We know that the value is stored in an unsigned 24 int
  // with kbps resolution. We expected it be represented as plus infinity.
  src.link_capacity_lower = DataRate::KilobitsPerSec(2 << 24);
  src.link_capacity_upper = DataRate::PlusInfinity();
  rtc::Buffer data = GetRemoteEstimateSerializer()->Serialize(src);

  NetworkStateEstimate dst;
  EXPECT_TRUE(GetRemoteEstimateSerializer()->Parse(data, &dst));
  EXPECT_TRUE(dst.link_capacity_lower.IsPlusInfinity());
  EXPECT_TRUE(dst.link_capacity_upper.IsPlusInfinity());
}

TEST(RemoteEstimateTest, DoesNotEncodeNegative) {
  NetworkStateEstimate src;
  src.link_capacity_lower = DataRate::MinusInfinity();
  src.link_capacity_upper = DataRate::MinusInfinity();
  rtc::Buffer data = GetRemoteEstimateSerializer()->Serialize(src);
  // Since MinusInfinity can't be represented, the buffer should be empty.
  EXPECT_EQ(data.size(), 0u);
  NetworkStateEstimate dst;
  dst.link_capacity_lower = DataRate::KilobitsPerSec(300);
  EXPECT_TRUE(GetRemoteEstimateSerializer()->Parse(data, &dst));
  // The fields will be left unchanged by the parser as they were not encoded.
  EXPECT_EQ(dst.link_capacity_lower, DataRate::KilobitsPerSec(300));
  EXPECT_TRUE(dst.link_capacity_upper.IsMinusInfinity());
}
}  // namespace rtcp
}  // namespace webrtc
