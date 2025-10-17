/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 11, 2024.
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
#include "audio/voip/voip_core.h"

#include "api/audio_codecs/builtin_audio_decoder_factory.h"
#include "api/audio_codecs/builtin_audio_encoder_factory.h"
#include "api/environment/environment_factory.h"
#include "modules/audio_device/include/mock_audio_device.h"
#include "modules/audio_processing/include/mock_audio_processing.h"
#include "test/gtest.h"
#include "test/mock_transport.h"
#include "test/run_loop.h"

namespace webrtc {
namespace {

using ::testing::NiceMock;
using ::testing::Return;

constexpr int kPcmuPayload = 0;
constexpr int kPcmuSampleRateHz = 8000;
constexpr int kDtmfEventDurationMs = 1000;
constexpr DtmfEvent kDtmfEventCode = DtmfEvent::kDigitZero;

class VoipCoreTest : public ::testing::Test {
 public:
  const SdpAudioFormat kPcmuFormat = {"pcmu", 8000, 1};

  VoipCoreTest() { audio_device_ = test::MockAudioDeviceModule::CreateNice(); }

  void SetUp() override {
    auto encoder_factory = CreateBuiltinAudioEncoderFactory();
    auto decoder_factory = CreateBuiltinAudioDecoderFactory();
    rtc::scoped_refptr<AudioProcessing> audio_processing =
        rtc::make_ref_counted<NiceMock<test::MockAudioProcessing>>();

    voip_core_ = std::make_unique<VoipCore>(
        CreateEnvironment(), std::move(encoder_factory),
        std::move(decoder_factory), audio_device_, std::move(audio_processing));
  }

  test::RunLoop run_loop_;
  std::unique_ptr<VoipCore> voip_core_;
  NiceMock<MockTransport> transport_;
  rtc::scoped_refptr<test::MockAudioDeviceModule> audio_device_;
};

// Validate expected API calls that involves with VoipCore. Some verification is
// involved with checking mock audio device.
TEST_F(VoipCoreTest, BasicVoipCoreOperation) {
  // Program mock as non-operational and ready to start.
  EXPECT_CALL(*audio_device_, Recording()).WillOnce(Return(false));
  EXPECT_CALL(*audio_device_, Playing()).WillOnce(Return(false));
  EXPECT_CALL(*audio_device_, InitRecording()).WillOnce(Return(0));
  EXPECT_CALL(*audio_device_, InitPlayout()).WillOnce(Return(0));
  EXPECT_CALL(*audio_device_, StartRecording()).WillOnce(Return(0));
  EXPECT_CALL(*audio_device_, StartPlayout()).WillOnce(Return(0));

  auto channel = voip_core_->CreateChannel(&transport_, 0xdeadc0de);

  EXPECT_EQ(voip_core_->SetSendCodec(channel, kPcmuPayload, kPcmuFormat),
            VoipResult::kOk);
  EXPECT_EQ(
      voip_core_->SetReceiveCodecs(channel, {{kPcmuPayload, kPcmuFormat}}),
      VoipResult::kOk);

  EXPECT_EQ(voip_core_->StartSend(channel), VoipResult::kOk);
  EXPECT_EQ(voip_core_->StartPlayout(channel), VoipResult::kOk);

  EXPECT_EQ(voip_core_->RegisterTelephoneEventType(channel, kPcmuPayload,
                                                   kPcmuSampleRateHz),
            VoipResult::kOk);

  EXPECT_EQ(
      voip_core_->SendDtmfEvent(channel, kDtmfEventCode, kDtmfEventDurationMs),
      VoipResult::kOk);

  // Program mock as operational that is ready to be stopped.
  EXPECT_CALL(*audio_device_, Recording()).WillOnce(Return(true));
  EXPECT_CALL(*audio_device_, Playing()).WillOnce(Return(true));
  EXPECT_CALL(*audio_device_, StopRecording()).WillOnce(Return(0));
  EXPECT_CALL(*audio_device_, StopPlayout()).WillOnce(Return(0));

  EXPECT_EQ(voip_core_->StopSend(channel), VoipResult::kOk);
  EXPECT_EQ(voip_core_->StopPlayout(channel), VoipResult::kOk);
  EXPECT_EQ(voip_core_->ReleaseChannel(channel), VoipResult::kOk);
}

TEST_F(VoipCoreTest, ExpectFailToUseReleasedChannelId) {
  auto channel = voip_core_->CreateChannel(&transport_, 0xdeadc0de);

  // Release right after creation.
  EXPECT_EQ(voip_core_->ReleaseChannel(channel), VoipResult::kOk);

  // Now use released channel.

  EXPECT_EQ(voip_core_->SetSendCodec(channel, kPcmuPayload, kPcmuFormat),
            VoipResult::kInvalidArgument);
  EXPECT_EQ(
      voip_core_->SetReceiveCodecs(channel, {{kPcmuPayload, kPcmuFormat}}),
      VoipResult::kInvalidArgument);
  EXPECT_EQ(voip_core_->RegisterTelephoneEventType(channel, kPcmuPayload,
                                                   kPcmuSampleRateHz),
            VoipResult::kInvalidArgument);
  EXPECT_EQ(voip_core_->StartSend(channel), VoipResult::kInvalidArgument);
  EXPECT_EQ(voip_core_->StartPlayout(channel), VoipResult::kInvalidArgument);
  EXPECT_EQ(
      voip_core_->SendDtmfEvent(channel, kDtmfEventCode, kDtmfEventDurationMs),
      VoipResult::kInvalidArgument);
}

TEST_F(VoipCoreTest, SendDtmfEventWithoutRegistering) {
  // Program mock as non-operational and ready to start send.
  EXPECT_CALL(*audio_device_, Recording()).WillOnce(Return(false));
  EXPECT_CALL(*audio_device_, InitRecording()).WillOnce(Return(0));
  EXPECT_CALL(*audio_device_, StartRecording()).WillOnce(Return(0));

  auto channel = voip_core_->CreateChannel(&transport_, 0xdeadc0de);

  EXPECT_EQ(voip_core_->SetSendCodec(channel, kPcmuPayload, kPcmuFormat),
            VoipResult::kOk);

  EXPECT_EQ(voip_core_->StartSend(channel), VoipResult::kOk);
  // Send Dtmf event without registering beforehand, thus payload
  // type is not set and kFailedPrecondition is expected.
  EXPECT_EQ(
      voip_core_->SendDtmfEvent(channel, kDtmfEventCode, kDtmfEventDurationMs),
      VoipResult::kFailedPrecondition);

  // Program mock as sending and is ready to be stopped.
  EXPECT_CALL(*audio_device_, Recording()).WillOnce(Return(true));
  EXPECT_CALL(*audio_device_, StopRecording()).WillOnce(Return(0));

  EXPECT_EQ(voip_core_->StopSend(channel), VoipResult::kOk);
  EXPECT_EQ(voip_core_->ReleaseChannel(channel), VoipResult::kOk);
}

TEST_F(VoipCoreTest, SendDtmfEventWithoutStartSend) {
  auto channel = voip_core_->CreateChannel(&transport_, 0xdeadc0de);

  EXPECT_EQ(voip_core_->RegisterTelephoneEventType(channel, kPcmuPayload,
                                                   kPcmuSampleRateHz),
            VoipResult::kOk);

  // Send Dtmf event without calling StartSend beforehand, thus
  // Dtmf events cannot be sent and kFailedPrecondition is expected.
  EXPECT_EQ(
      voip_core_->SendDtmfEvent(channel, kDtmfEventCode, kDtmfEventDurationMs),
      VoipResult::kFailedPrecondition);

  EXPECT_EQ(voip_core_->ReleaseChannel(channel), VoipResult::kOk);
}

TEST_F(VoipCoreTest, StartSendAndPlayoutWithoutSettingCodec) {
  auto channel = voip_core_->CreateChannel(&transport_, 0xdeadc0de);

  // Call StartSend and StartPlayout without setting send/receive
  // codec. Code should see that codecs aren't set and return false.
  EXPECT_EQ(voip_core_->StartSend(channel), VoipResult::kFailedPrecondition);
  EXPECT_EQ(voip_core_->StartPlayout(channel), VoipResult::kFailedPrecondition);

  EXPECT_EQ(voip_core_->ReleaseChannel(channel), VoipResult::kOk);
}

TEST_F(VoipCoreTest, StopSendAndPlayoutWithoutStarting) {
  auto channel = voip_core_->CreateChannel(&transport_, 0xdeadc0de);

  EXPECT_EQ(voip_core_->SetSendCodec(channel, kPcmuPayload, kPcmuFormat),
            VoipResult::kOk);
  EXPECT_EQ(
      voip_core_->SetReceiveCodecs(channel, {{kPcmuPayload, kPcmuFormat}}),
      VoipResult::kOk);

  // Call StopSend and StopPlayout without starting them in
  // the first place. Should see that it is already in the
  // stopped state and return true.
  EXPECT_EQ(voip_core_->StopSend(channel), VoipResult::kOk);
  EXPECT_EQ(voip_core_->StopPlayout(channel), VoipResult::kOk);

  EXPECT_EQ(voip_core_->ReleaseChannel(channel), VoipResult::kOk);
}

}  // namespace
}  // namespace webrtc
