/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 21, 2023.
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
#include "modules/rtp_rtcp/source/rtcp_packet/rrtr.h"

#include "test/gtest.h"

using webrtc::rtcp::Rrtr;

namespace webrtc {
namespace {

const uint32_t kNtpSec = 0x12345678;
const uint32_t kNtpFrac = 0x23456789;
const uint8_t kBlock[] = {0x04, 0x00, 0x00, 0x02, 0x12, 0x34,
                          0x56, 0x78, 0x23, 0x45, 0x67, 0x89};
const size_t kBlockSizeBytes = sizeof(kBlock);
static_assert(
    kBlockSizeBytes == Rrtr::kLength,
    "Size of manually created Rrtr block should match class constant");

TEST(RtcpPacketRrtrTest, Create) {
  uint8_t buffer[Rrtr::kLength];
  Rrtr rrtr;
  rrtr.SetNtp(NtpTime(kNtpSec, kNtpFrac));

  rrtr.Create(buffer);
  EXPECT_EQ(0, memcmp(buffer, kBlock, kBlockSizeBytes));
}

TEST(RtcpPacketRrtrTest, Parse) {
  Rrtr read_rrtr;
  read_rrtr.Parse(kBlock);

  // Run checks on const object to ensure all accessors have const modifier.
  const Rrtr& parsed = read_rrtr;

  EXPECT_EQ(kNtpSec, parsed.ntp().seconds());
  EXPECT_EQ(kNtpFrac, parsed.ntp().fractions());
}

}  // namespace
}  // namespace webrtc
