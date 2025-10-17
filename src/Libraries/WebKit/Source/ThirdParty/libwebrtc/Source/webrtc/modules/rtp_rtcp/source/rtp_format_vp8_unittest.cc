/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 15, 2023.
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
#include "modules/rtp_rtcp/source/rtp_format_vp8.h"

#include <memory>

#include "modules/rtp_rtcp/source/rtp_format_vp8_test_helper.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

constexpr RtpPacketizer::PayloadSizeLimits kNoSizeLimits;

TEST(RtpPacketizerVp8Test, EmptyPayload) {
  RTPVideoHeaderVP8 hdr_info;
  hdr_info.InitRTPVideoHeaderVP8();
  hdr_info.pictureId = 200;
  RtpFormatVp8TestHelper helper(&hdr_info, /*payload_len=*/30);

  RtpPacketizer::PayloadSizeLimits limits;
  limits.max_payload_len = 12;  // Small enough to produce 4 packets.
  RtpPacketizerVp8 packetizer({}, limits, hdr_info);
  EXPECT_EQ(packetizer.NumPackets(), 0u);
}

TEST(RtpPacketizerVp8Test, ResultPacketsAreAlmostEqualSize) {
  RTPVideoHeaderVP8 hdr_info;
  hdr_info.InitRTPVideoHeaderVP8();
  hdr_info.pictureId = 200;
  RtpFormatVp8TestHelper helper(&hdr_info, /*payload_len=*/30);

  RtpPacketizer::PayloadSizeLimits limits;
  limits.max_payload_len = 12;  // Small enough to produce 4 packets.
  RtpPacketizerVp8 packetizer(helper.payload(), limits, hdr_info);

  const size_t kExpectedSizes[] = {11, 11, 12, 12};
  helper.GetAllPacketsAndCheck(&packetizer, kExpectedSizes);
}

TEST(RtpPacketizerVp8Test, EqualSizeWithLastPacketReduction) {
  RTPVideoHeaderVP8 hdr_info;
  hdr_info.InitRTPVideoHeaderVP8();
  hdr_info.pictureId = 200;
  RtpFormatVp8TestHelper helper(&hdr_info, /*payload_len=*/43);

  RtpPacketizer::PayloadSizeLimits limits;
  limits.max_payload_len = 15;  // Small enough to produce 5 packets.
  limits.last_packet_reduction_len = 5;
  RtpPacketizerVp8 packetizer(helper.payload(), limits, hdr_info);

  // Calculated by hand. VP8 payload descriptors are 4 byte each. 5 packets is
  // minimum possible to fit 43 payload bytes into packets with capacity of
  // 15 - 4 = 11 and leave 5 free bytes in the last packet. All packets are
  // almost equal in size, even last packet if counted with free space (which
  // will be filled up the stack by extra long RTP header).
  const size_t kExpectedSizes[] = {13, 13, 14, 14, 9};
  helper.GetAllPacketsAndCheck(&packetizer, kExpectedSizes);
}

// Verify that non-reference bit is set.
TEST(RtpPacketizerVp8Test, NonReferenceBit) {
  RTPVideoHeaderVP8 hdr_info;
  hdr_info.InitRTPVideoHeaderVP8();
  hdr_info.nonReference = true;
  RtpFormatVp8TestHelper helper(&hdr_info, /*payload_len=*/30);

  RtpPacketizer::PayloadSizeLimits limits;
  limits.max_payload_len = 25;  // Small enough to produce two packets.
  RtpPacketizerVp8 packetizer(helper.payload(), limits, hdr_info);

  const size_t kExpectedSizes[] = {16, 16};
  helper.GetAllPacketsAndCheck(&packetizer, kExpectedSizes);
}

// Verify Tl0PicIdx and TID fields, and layerSync bit.
TEST(RtpPacketizerVp8Test, Tl0PicIdxAndTID) {
  RTPVideoHeaderVP8 hdr_info;
  hdr_info.InitRTPVideoHeaderVP8();
  hdr_info.tl0PicIdx = 117;
  hdr_info.temporalIdx = 2;
  hdr_info.layerSync = true;
  RtpFormatVp8TestHelper helper(&hdr_info, /*payload_len=*/30);

  RtpPacketizerVp8 packetizer(helper.payload(), kNoSizeLimits, hdr_info);

  const size_t kExpectedSizes[1] = {helper.payload_size() + 4};
  helper.GetAllPacketsAndCheck(&packetizer, kExpectedSizes);
}

TEST(RtpPacketizerVp8Test, KeyIdx) {
  RTPVideoHeaderVP8 hdr_info;
  hdr_info.InitRTPVideoHeaderVP8();
  hdr_info.keyIdx = 17;
  RtpFormatVp8TestHelper helper(&hdr_info, /*payload_len=*/30);

  RtpPacketizerVp8 packetizer(helper.payload(), kNoSizeLimits, hdr_info);

  const size_t kExpectedSizes[1] = {helper.payload_size() + 3};
  helper.GetAllPacketsAndCheck(&packetizer, kExpectedSizes);
}

// Verify TID field and KeyIdx field in combination.
TEST(RtpPacketizerVp8Test, TIDAndKeyIdx) {
  RTPVideoHeaderVP8 hdr_info;
  hdr_info.InitRTPVideoHeaderVP8();
  hdr_info.temporalIdx = 1;
  hdr_info.keyIdx = 5;
  RtpFormatVp8TestHelper helper(&hdr_info, /*payload_len=*/30);

  RtpPacketizerVp8 packetizer(helper.payload(), kNoSizeLimits, hdr_info);

  const size_t kExpectedSizes[1] = {helper.payload_size() + 3};
  helper.GetAllPacketsAndCheck(&packetizer, kExpectedSizes);
}

}  // namespace
}  // namespace webrtc
