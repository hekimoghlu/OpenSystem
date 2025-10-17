/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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
// Integration tests for PeerConnection.
// These tests exercise a full stack for the SVC extension.

#include <stdint.h>

#include <functional>
#include <vector>

#include "absl/strings/match.h"
#include "api/rtc_error.h"
#include "api/rtp_parameters.h"
#include "api/rtp_transceiver_interface.h"
#include "api/scoped_refptr.h"
#include "pc/test/integration_test_helpers.h"
#include "rtc_base/crypto_random.h"
#include "rtc_base/gunit.h"
#include "test/gtest.h"

namespace webrtc {

namespace {

class PeerConnectionSVCIntegrationTest
    : public PeerConnectionIntegrationBaseTest {
 protected:
  PeerConnectionSVCIntegrationTest()
      : PeerConnectionIntegrationBaseTest(SdpSemantics::kUnifiedPlan) {}

  RTCError SetCodecPreferences(
      rtc::scoped_refptr<RtpTransceiverInterface> transceiver,
      absl::string_view codec_name) {
    RtpCapabilities capabilities =
        caller()->pc_factory()->GetRtpReceiverCapabilities(
            cricket::MEDIA_TYPE_VIDEO);
    std::vector<RtpCodecCapability> codecs;
    for (const RtpCodecCapability& codec_capability : capabilities.codecs) {
      if (codec_capability.name == codec_name)
        codecs.push_back(codec_capability);
    }
    return transceiver->SetCodecPreferences(codecs);
  }
};

TEST_F(PeerConnectionSVCIntegrationTest, AddTransceiverAcceptsL1T1) {
  ASSERT_TRUE(CreatePeerConnectionWrappers());
  ConnectFakeSignaling();
  RtpTransceiverInit init;
  RtpEncodingParameters encoding_parameters;
  encoding_parameters.scalability_mode = "L1T1";
  init.send_encodings.push_back(encoding_parameters);
  auto transceiver_or_error =
      caller()->pc()->AddTransceiver(caller()->CreateLocalVideoTrack(), init);
  EXPECT_TRUE(transceiver_or_error.ok());
}

TEST_F(PeerConnectionSVCIntegrationTest, AddTransceiverAcceptsL3T3) {
  ASSERT_TRUE(CreatePeerConnectionWrappers());
  ConnectFakeSignaling();
  RtpTransceiverInit init;
  RtpEncodingParameters encoding_parameters;
  encoding_parameters.scalability_mode = "L3T3";
  init.send_encodings.push_back(encoding_parameters);
  auto transceiver_or_error =
      caller()->pc()->AddTransceiver(caller()->CreateLocalVideoTrack(), init);
  EXPECT_TRUE(transceiver_or_error.ok());
}

TEST_F(PeerConnectionSVCIntegrationTest,
       AddTransceiverRejectsUnknownScalabilityMode) {
  ASSERT_TRUE(CreatePeerConnectionWrappers());
  ConnectFakeSignaling();
  RtpTransceiverInit init;
  RtpEncodingParameters encoding_parameters;
  encoding_parameters.scalability_mode = "FOOBAR";
  init.send_encodings.push_back(encoding_parameters);
  auto transceiver_or_error =
      caller()->pc()->AddTransceiver(caller()->CreateLocalVideoTrack(), init);
  EXPECT_FALSE(transceiver_or_error.ok());
  EXPECT_EQ(transceiver_or_error.error().type(),
            RTCErrorType::UNSUPPORTED_OPERATION);
}

TEST_F(PeerConnectionSVCIntegrationTest, SetParametersAcceptsL1T3WithVP8) {
  ASSERT_TRUE(CreatePeerConnectionWrappers());
  ConnectFakeSignaling();

  RtpCapabilities capabilities =
      caller()->pc_factory()->GetRtpReceiverCapabilities(
          cricket::MEDIA_TYPE_VIDEO);
  std::vector<RtpCodecCapability> vp8_codec;
  for (const RtpCodecCapability& codec_capability : capabilities.codecs) {
    if (codec_capability.name == cricket::kVp8CodecName)
      vp8_codec.push_back(codec_capability);
  }

  RtpTransceiverInit init;
  RtpEncodingParameters encoding_parameters;
  init.send_encodings.push_back(encoding_parameters);
  auto transceiver_or_error =
      caller()->pc()->AddTransceiver(caller()->CreateLocalVideoTrack(), init);
  ASSERT_TRUE(transceiver_or_error.ok());
  auto transceiver = transceiver_or_error.MoveValue();
  EXPECT_TRUE(transceiver->SetCodecPreferences(vp8_codec).ok());

  RtpParameters parameters = transceiver->sender()->GetParameters();
  ASSERT_EQ(parameters.encodings.size(), 1u);
  parameters.encodings[0].scalability_mode = "L1T3";
  auto result = transceiver->sender()->SetParameters(parameters);
  EXPECT_TRUE(result.ok());
}

TEST_F(PeerConnectionSVCIntegrationTest,
       SetParametersAcceptsL1T3WithVP8AfterNegotiation) {
  ASSERT_TRUE(CreatePeerConnectionWrappers());
  ConnectFakeSignaling();

  RtpTransceiverInit init;
  RtpEncodingParameters encoding_parameters;
  init.send_encodings.push_back(encoding_parameters);
  auto transceiver_or_error =
      caller()->pc()->AddTransceiver(caller()->CreateLocalVideoTrack(), init);
  ASSERT_TRUE(transceiver_or_error.ok());
  auto transceiver = transceiver_or_error.MoveValue();
  EXPECT_TRUE(SetCodecPreferences(transceiver, cricket::kVp8CodecName).ok());

  caller()->CreateAndSetAndSignalOffer();
  ASSERT_TRUE_WAIT(SignalingStateStable(), kDefaultTimeout);

  RtpParameters parameters = transceiver->sender()->GetParameters();
  ASSERT_EQ(parameters.encodings.size(), 1u);
  parameters.encodings[0].scalability_mode = "L1T3";
  auto result = transceiver->sender()->SetParameters(parameters);
  EXPECT_TRUE(result.ok());
}

TEST_F(PeerConnectionSVCIntegrationTest,
       SetParametersAcceptsL3T3WithVP9AfterNegotiation) {
  ASSERT_TRUE(CreatePeerConnectionWrappers());
  ConnectFakeSignaling();

  RtpTransceiverInit init;
  RtpEncodingParameters encoding_parameters;
  init.send_encodings.push_back(encoding_parameters);
  auto transceiver_or_error =
      caller()->pc()->AddTransceiver(caller()->CreateLocalVideoTrack(), init);
  ASSERT_TRUE(transceiver_or_error.ok());
  auto transceiver = transceiver_or_error.MoveValue();
  EXPECT_TRUE(SetCodecPreferences(transceiver, cricket::kVp9CodecName).ok());

  caller()->CreateAndSetAndSignalOffer();
  ASSERT_TRUE_WAIT(SignalingStateStable(), kDefaultTimeout);

  RtpParameters parameters = transceiver->sender()->GetParameters();
  ASSERT_EQ(parameters.encodings.size(), 1u);
  parameters.encodings[0].scalability_mode = "L3T3";
  auto result = transceiver->sender()->SetParameters(parameters);
  EXPECT_TRUE(result.ok());
}

TEST_F(PeerConnectionSVCIntegrationTest,
       SetParametersRejectsL3T3WithVP8AfterNegotiation) {
  ASSERT_TRUE(CreatePeerConnectionWrappers());
  ConnectFakeSignaling();

  RtpTransceiverInit init;
  RtpEncodingParameters encoding_parameters;
  init.send_encodings.push_back(encoding_parameters);
  auto transceiver_or_error =
      caller()->pc()->AddTransceiver(caller()->CreateLocalVideoTrack(), init);
  ASSERT_TRUE(transceiver_or_error.ok());
  auto transceiver = transceiver_or_error.MoveValue();
  EXPECT_TRUE(SetCodecPreferences(transceiver, cricket::kVp8CodecName).ok());

  caller()->CreateAndSetAndSignalOffer();
  ASSERT_TRUE_WAIT(SignalingStateStable(), kDefaultTimeout);

  RtpParameters parameters = transceiver->sender()->GetParameters();
  ASSERT_EQ(parameters.encodings.size(), 1u);
  parameters.encodings[0].scalability_mode = "L3T3";
  auto result = transceiver->sender()->SetParameters(parameters);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.type(), RTCErrorType::INVALID_MODIFICATION);
}

TEST_F(PeerConnectionSVCIntegrationTest,
       SetParametersRejectsInvalidModeWithVP9AfterNegotiation) {
  ASSERT_TRUE(CreatePeerConnectionWrappers());
  ConnectFakeSignaling();

  RtpTransceiverInit init;
  RtpEncodingParameters encoding_parameters;
  init.send_encodings.push_back(encoding_parameters);
  auto transceiver_or_error =
      caller()->pc()->AddTransceiver(caller()->CreateLocalVideoTrack(), init);
  ASSERT_TRUE(transceiver_or_error.ok());
  auto transceiver = transceiver_or_error.MoveValue();
  EXPECT_TRUE(SetCodecPreferences(transceiver, cricket::kVp9CodecName).ok());

  caller()->CreateAndSetAndSignalOffer();
  ASSERT_TRUE_WAIT(SignalingStateStable(), kDefaultTimeout);

  RtpParameters parameters = transceiver->sender()->GetParameters();
  ASSERT_EQ(parameters.encodings.size(), 1u);
  parameters.encodings[0].scalability_mode = "FOOBAR";
  auto result = transceiver->sender()->SetParameters(parameters);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.type(), RTCErrorType::INVALID_MODIFICATION);
}

TEST_F(PeerConnectionSVCIntegrationTest, FallbackToL1Tx) {
  ASSERT_TRUE(CreatePeerConnectionWrappers());
  ConnectFakeSignaling();

  RtpTransceiverInit init;
  RtpEncodingParameters encoding_parameters;
  init.send_encodings.push_back(encoding_parameters);
  auto transceiver_or_error =
      caller()->pc()->AddTransceiver(caller()->CreateLocalVideoTrack(), init);
  ASSERT_TRUE(transceiver_or_error.ok());
  auto caller_transceiver = transceiver_or_error.MoveValue();

  RtpCapabilities capabilities =
      caller()->pc_factory()->GetRtpReceiverCapabilities(
          cricket::MEDIA_TYPE_VIDEO);
  std::vector<RtpCodecCapability> send_codecs = capabilities.codecs;
  // Only keep VP9 in the caller
  send_codecs.erase(std::partition(send_codecs.begin(), send_codecs.end(),
                                   [](const auto& codec) -> bool {
                                     return codec.name ==
                                            cricket::kVp9CodecName;
                                   }),
                    send_codecs.end());
  ASSERT_FALSE(send_codecs.empty());
  caller_transceiver->SetCodecPreferences(send_codecs);

  // L3T3 should be supported by VP9
  RtpParameters parameters = caller_transceiver->sender()->GetParameters();
  ASSERT_EQ(parameters.encodings.size(), 1u);
  parameters.encodings[0].scalability_mode = "L3T3";
  auto result = caller_transceiver->sender()->SetParameters(parameters);
  EXPECT_TRUE(result.ok());

  caller()->CreateAndSetAndSignalOffer();
  ASSERT_TRUE_WAIT(SignalingStateStable(), kDefaultTimeout);

  parameters = caller_transceiver->sender()->GetParameters();
  ASSERT_TRUE(parameters.encodings[0].scalability_mode.has_value());
  EXPECT_TRUE(
      absl::StartsWith(*parameters.encodings[0].scalability_mode, "L3T3"));

  // Keep only VP8 in the caller
  send_codecs = capabilities.codecs;
  send_codecs.erase(std::partition(send_codecs.begin(), send_codecs.end(),
                                   [](const auto& codec) -> bool {
                                     return codec.name ==
                                            cricket::kVp8CodecName;
                                   }),
                    send_codecs.end());
  ASSERT_FALSE(send_codecs.empty());
  caller_transceiver->SetCodecPreferences(send_codecs);

  // Renegotiate to force the new codec list to be used
  caller()->CreateAndSetAndSignalOffer();
  ASSERT_TRUE_WAIT(SignalingStateStable(), kDefaultTimeout);

  // Fallback should happen and L3T3 is not used anymore
  parameters = caller_transceiver->sender()->GetParameters();
  ASSERT_TRUE(parameters.encodings[0].scalability_mode.has_value());
  EXPECT_TRUE(
      absl::StartsWith(*parameters.encodings[0].scalability_mode, "L1T"));
}
}  // namespace

}  // namespace webrtc
