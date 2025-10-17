/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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
// This file contains tests that verify that congestion control options
// are correctly negotiated in the SDP offer/answer.

#include <string>

#include "absl/strings/str_cat.h"
#include "api/peer_connection_interface.h"
#include "pc/test/integration_test_helpers.h"
#include "rtc_base/gunit.h"
#include "test/field_trial.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {

using testing::HasSubstr;

class PeerConnectionCongestionControlTest
    : public PeerConnectionIntegrationBaseTest {
 public:
  PeerConnectionCongestionControlTest()
      : PeerConnectionIntegrationBaseTest(SdpSemantics::kUnifiedPlan) {}
};

TEST_F(PeerConnectionCongestionControlTest, OfferContainsCcfbIfEnabled) {
  test::ScopedFieldTrials trials(
      "WebRTC-RFC8888CongestionControlFeedback/Enabled/");
  ASSERT_TRUE(CreatePeerConnectionWrappers());
  caller()->AddAudioVideoTracks();
  auto offer = caller()->CreateOfferAndWait();
  std::string offer_str = absl::StrCat(*offer);
  EXPECT_THAT(offer_str, HasSubstr("a=rtcp-fb:* ack ccfb\r\n"));
}

TEST_F(PeerConnectionCongestionControlTest, ReceiveOfferSetsCcfbFlag) {
  test::ScopedFieldTrials trials(
      "WebRTC-RFC8888CongestionControlFeedback/Enabled/");
  ASSERT_TRUE(CreatePeerConnectionWrappers());
  ConnectFakeSignalingForSdpOnly();
  caller()->AddAudioVideoTracks();
  caller()->CreateAndSetAndSignalOffer();
  ASSERT_TRUE_WAIT(SignalingStateStable(), kDefaultTimeout);
  // Check that the callee parsed it.
  auto parsed_contents =
      callee()->pc()->remote_description()->description()->contents();
  EXPECT_FALSE(parsed_contents.empty());
  for (const auto& content : parsed_contents) {
    EXPECT_TRUE(content.media_description()->rtcp_fb_ack_ccfb());
  }
  // Check that the caller also parsed it.
  parsed_contents =
      caller()->pc()->remote_description()->description()->contents();
  EXPECT_FALSE(parsed_contents.empty());
  for (const auto& content : parsed_contents) {
    EXPECT_TRUE(content.media_description()->rtcp_fb_ack_ccfb());
  }
}

}  // namespace webrtc
