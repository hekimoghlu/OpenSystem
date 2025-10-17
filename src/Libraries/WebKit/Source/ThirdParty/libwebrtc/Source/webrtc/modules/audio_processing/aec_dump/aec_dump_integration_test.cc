/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 15, 2024.
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
#include <array>
#include <memory>
#include <utility>

#include "api/audio/audio_processing.h"
#include "api/audio/builtin_audio_processing_builder.h"
#include "api/environment/environment_factory.h"
#include "modules/audio_processing/aec_dump/mock_aec_dump.h"
#include "modules/audio_processing/audio_processing_impl.h"

using ::testing::_;
using ::testing::AtLeast;
using ::testing::Exactly;
using ::testing::StrictMock;

namespace {
webrtc::scoped_refptr<webrtc::AudioProcessing> CreateAudioProcessing() {
  webrtc::scoped_refptr<webrtc::AudioProcessing> apm =
      webrtc::BuiltinAudioProcessingBuilder().Build(
          webrtc::CreateEnvironment());
  RTC_DCHECK(apm);
  return apm;
}

std::unique_ptr<webrtc::test::MockAecDump> CreateMockAecDump() {
  auto mock_aec_dump =
      std::make_unique<testing::StrictMock<webrtc::test::MockAecDump>>();
  EXPECT_CALL(*mock_aec_dump.get(), WriteConfig(_)).Times(AtLeast(1));
  EXPECT_CALL(*mock_aec_dump.get(), WriteInitMessage(_, _)).Times(AtLeast(1));
  return std::unique_ptr<webrtc::test::MockAecDump>(std::move(mock_aec_dump));
}

}  // namespace

TEST(AecDumpIntegration, ConfigurationAndInitShouldBeLogged) {
  auto apm = CreateAudioProcessing();

  apm->AttachAecDump(CreateMockAecDump());
}

TEST(AecDumpIntegration,
     RenderStreamShouldBeLoggedOnceEveryProcessReverseStream) {
  auto apm = CreateAudioProcessing();
  auto mock_aec_dump = CreateMockAecDump();
  constexpr int kNumChannels = 1;
  constexpr int kNumSampleRateHz = 16000;
  constexpr int kNumSamplesPerChannel = kNumSampleRateHz / 100;
  std::array<int16_t, kNumSamplesPerChannel * kNumChannels> frame;
  frame.fill(0.f);
  webrtc::StreamConfig stream_config(kNumSampleRateHz, kNumChannels);

  EXPECT_CALL(*mock_aec_dump.get(), WriteRenderStreamMessage(_, _, _))
      .Times(Exactly(1));

  apm->AttachAecDump(std::move(mock_aec_dump));
  apm->ProcessReverseStream(frame.data(), stream_config, stream_config,
                            frame.data());
}

TEST(AecDumpIntegration, CaptureStreamShouldBeLoggedOnceEveryProcessStream) {
  auto apm = CreateAudioProcessing();
  auto mock_aec_dump = CreateMockAecDump();
  constexpr int kNumChannels = 1;
  constexpr int kNumSampleRateHz = 16000;
  constexpr int kNumSamplesPerChannel = kNumSampleRateHz / 100;
  std::array<int16_t, kNumSamplesPerChannel * kNumChannels> frame;
  frame.fill(0.f);

  webrtc::StreamConfig stream_config(kNumSampleRateHz, kNumChannels);

  EXPECT_CALL(*mock_aec_dump.get(), AddCaptureStreamInput(_, _, _))
      .Times(AtLeast(1));

  EXPECT_CALL(*mock_aec_dump.get(), AddCaptureStreamOutput(_, _, _))
      .Times(Exactly(1));

  EXPECT_CALL(*mock_aec_dump.get(), AddAudioProcessingState(_))
      .Times(Exactly(1));

  EXPECT_CALL(*mock_aec_dump.get(), WriteCaptureStreamMessage())
      .Times(Exactly(1));

  apm->AttachAecDump(std::move(mock_aec_dump));
  apm->ProcessStream(frame.data(), stream_config, stream_config, frame.data());
}
