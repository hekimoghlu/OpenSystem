/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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
#include "modules/rtp_rtcp/source/rtcp_packet.h"

#include "modules/rtp_rtcp/source/rtcp_packet/receiver_report.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace {

using ::testing::_;
using ::testing::MockFunction;
using ::webrtc::rtcp::ReceiverReport;
using ::webrtc::rtcp::ReportBlock;

const uint32_t kSenderSsrc = 0x12345678;

TEST(RtcpPacketTest, BuildWithTooSmallBuffer) {
  ReportBlock rb;
  ReceiverReport rr;
  rr.SetSenderSsrc(kSenderSsrc);
  EXPECT_TRUE(rr.AddReportBlock(rb));

  const size_t kRrLength = 8;
  const size_t kReportBlockLength = 24;

  // No packet.
  MockFunction<void(rtc::ArrayView<const uint8_t>)> callback;
  EXPECT_CALL(callback, Call(_)).Times(0);
  const size_t kBufferSize = kRrLength + kReportBlockLength - 1;
  EXPECT_FALSE(rr.Build(kBufferSize, callback.AsStdFunction()));
}

}  // namespace
