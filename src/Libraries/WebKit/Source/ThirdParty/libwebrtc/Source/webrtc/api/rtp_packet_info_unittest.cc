/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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
#include "api/rtp_packet_info.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "api/rtp_headers.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "test/gtest.h"

namespace webrtc {

TEST(RtpPacketInfoTest, Ssrc) {
  constexpr uint32_t kValue = 4038189233;

  RtpPacketInfo lhs;
  RtpPacketInfo rhs;

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);

  rhs.set_ssrc(kValue);
  EXPECT_EQ(rhs.ssrc(), kValue);

  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);

  lhs = rhs;

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);

  rhs = RtpPacketInfo();
  EXPECT_NE(rhs.ssrc(), kValue);

  rhs = RtpPacketInfo(/*ssrc=*/kValue, /*csrcs=*/{}, /*rtp_timestamp=*/{},
                      /*receive_time=*/Timestamp::Zero());
  EXPECT_EQ(rhs.ssrc(), kValue);
}

TEST(RtpPacketInfoTest, Csrcs) {
  const std::vector<uint32_t> value = {4038189233, 3016333617, 1207992985};

  RtpPacketInfo lhs;
  RtpPacketInfo rhs;

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);

  rhs.set_csrcs(value);
  EXPECT_EQ(rhs.csrcs(), value);

  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);

  lhs = rhs;

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);

  rhs = RtpPacketInfo();
  EXPECT_NE(rhs.csrcs(), value);

  rhs = RtpPacketInfo(/*ssrc=*/{}, /*csrcs=*/value, /*rtp_timestamp=*/{},
                      /*receive_time=*/Timestamp::Zero());
  EXPECT_EQ(rhs.csrcs(), value);
}

TEST(RtpPacketInfoTest, RtpTimestamp) {
  constexpr uint32_t kValue = 4038189233;

  RtpPacketInfo lhs;
  RtpPacketInfo rhs;

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);

  rhs.set_rtp_timestamp(kValue);
  EXPECT_EQ(rhs.rtp_timestamp(), kValue);

  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);

  lhs = rhs;

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);

  rhs = RtpPacketInfo();
  EXPECT_NE(rhs.rtp_timestamp(), kValue);

  rhs = RtpPacketInfo(/*ssrc=*/{}, /*csrcs=*/{}, /*rtp_timestamp=*/kValue,
                      /*receive_time=*/Timestamp::Zero());
  EXPECT_EQ(rhs.rtp_timestamp(), kValue);
}

TEST(RtpPacketInfoTest, ReceiveTimeMs) {
  constexpr Timestamp kValue = Timestamp::Micros(8868963877546349045LL);

  RtpPacketInfo lhs;
  RtpPacketInfo rhs;

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);

  rhs.set_receive_time(kValue);
  EXPECT_EQ(rhs.receive_time(), kValue);

  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);

  lhs = rhs;

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);

  rhs = RtpPacketInfo();
  EXPECT_NE(rhs.receive_time(), kValue);

  rhs = RtpPacketInfo(/*ssrc=*/{}, /*csrcs=*/{}, /*rtp_timestamp=*/{},
                      /*receive_time=*/kValue);
  EXPECT_EQ(rhs.receive_time(), kValue);
}

TEST(RtpPacketInfoTest, AudioLevel) {
  constexpr std::optional<uint8_t> kValue = 31;

  RtpPacketInfo lhs;
  RtpPacketInfo rhs;

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);

  rhs.set_audio_level(kValue);
  EXPECT_EQ(rhs.audio_level(), kValue);

  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);

  lhs = rhs;

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);

  rhs = RtpPacketInfo();
  EXPECT_NE(rhs.audio_level(), kValue);

  rhs = RtpPacketInfo(/*ssrc=*/{}, /*csrcs=*/{}, /*rtp_timestamp=*/{},
                      /*receive_time=*/Timestamp::Zero());
  rhs.set_audio_level(kValue);
  EXPECT_EQ(rhs.audio_level(), kValue);
}

TEST(RtpPacketInfoTest, AbsoluteCaptureTime) {
  constexpr std::optional<AbsoluteCaptureTime> kValue = AbsoluteCaptureTime{
      .absolute_capture_timestamp = 12, .estimated_capture_clock_offset = 34};

  RtpPacketInfo lhs;
  RtpPacketInfo rhs;

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);

  rhs.set_absolute_capture_time(kValue);
  EXPECT_EQ(rhs.absolute_capture_time(), kValue);

  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);

  lhs = rhs;

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);

  rhs = RtpPacketInfo();
  EXPECT_NE(rhs.absolute_capture_time(), kValue);

  rhs = RtpPacketInfo(/*ssrc=*/{}, /*csrcs=*/{}, /*rtp_timestamp=*/{},
                      /*receive_time=*/Timestamp::Zero());
  rhs.set_absolute_capture_time(kValue);
  EXPECT_EQ(rhs.absolute_capture_time(), kValue);
}

TEST(RtpPacketInfoTest, LocalCaptureClockOffset) {
  constexpr TimeDelta kValue = TimeDelta::Micros(8868963877546349045LL);

  RtpPacketInfo lhs;
  RtpPacketInfo rhs;

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);

  rhs.set_local_capture_clock_offset(kValue);
  EXPECT_EQ(rhs.local_capture_clock_offset(), kValue);

  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);

  lhs = rhs;

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);

  rhs = RtpPacketInfo();
  EXPECT_EQ(rhs.local_capture_clock_offset(), std::nullopt);

  rhs = RtpPacketInfo(/*ssrc=*/{}, /*csrcs=*/{}, /*rtp_timestamp=*/{},
                      /*receive_time=*/Timestamp::Zero());
  rhs.set_local_capture_clock_offset(kValue);
  EXPECT_EQ(rhs.local_capture_clock_offset(), kValue);
}

}  // namespace webrtc
