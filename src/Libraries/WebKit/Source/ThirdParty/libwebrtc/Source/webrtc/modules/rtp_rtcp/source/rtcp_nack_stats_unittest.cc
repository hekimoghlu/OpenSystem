/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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
#include "modules/rtp_rtcp/source/rtcp_nack_stats.h"

#include "test/gtest.h"

namespace webrtc {

TEST(RtcpNackStatsTest, Requests) {
  RtcpNackStats stats;
  EXPECT_EQ(0U, stats.unique_requests());
  EXPECT_EQ(0U, stats.requests());
  stats.ReportRequest(10);
  EXPECT_EQ(1U, stats.unique_requests());
  EXPECT_EQ(1U, stats.requests());

  stats.ReportRequest(10);
  EXPECT_EQ(1U, stats.unique_requests());
  stats.ReportRequest(11);
  EXPECT_EQ(2U, stats.unique_requests());

  stats.ReportRequest(11);
  EXPECT_EQ(2U, stats.unique_requests());
  stats.ReportRequest(13);
  EXPECT_EQ(3U, stats.unique_requests());

  stats.ReportRequest(11);
  EXPECT_EQ(3U, stats.unique_requests());
  EXPECT_EQ(6U, stats.requests());
}

TEST(RtcpNackStatsTest, RequestsWithWrap) {
  RtcpNackStats stats;
  stats.ReportRequest(65534);
  EXPECT_EQ(1U, stats.unique_requests());

  stats.ReportRequest(65534);
  EXPECT_EQ(1U, stats.unique_requests());
  stats.ReportRequest(65535);
  EXPECT_EQ(2U, stats.unique_requests());

  stats.ReportRequest(65535);
  EXPECT_EQ(2U, stats.unique_requests());
  stats.ReportRequest(0);
  EXPECT_EQ(3U, stats.unique_requests());

  stats.ReportRequest(65535);
  EXPECT_EQ(3U, stats.unique_requests());
  stats.ReportRequest(0);
  EXPECT_EQ(3U, stats.unique_requests());
  stats.ReportRequest(1);
  EXPECT_EQ(4U, stats.unique_requests());
  EXPECT_EQ(8U, stats.requests());
}

}  // namespace webrtc
