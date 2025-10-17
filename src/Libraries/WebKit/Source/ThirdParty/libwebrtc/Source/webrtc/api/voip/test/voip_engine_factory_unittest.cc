/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 31, 2023.
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
#include "api/voip/voip_engine_factory.h"

#include <memory>
#include <utility>

#include "api/make_ref_counted.h"
#include "api/task_queue/default_task_queue_factory.h"
#include "modules/audio_device/include/mock_audio_device.h"
#include "modules/audio_processing/include/mock_audio_processing.h"
#include "test/gmock.h"
#include "test/gtest.h"
#include "test/mock_audio_decoder_factory.h"
#include "test/mock_audio_encoder_factory.h"

namespace webrtc {
namespace {

using ::testing::NiceMock;

// Create voip engine with mock modules as normal use case.
TEST(VoipEngineFactoryTest, CreateEngineWithMockModules) {
  VoipEngineConfig config;
  config.encoder_factory = rtc::make_ref_counted<MockAudioEncoderFactory>();
  config.decoder_factory = rtc::make_ref_counted<MockAudioDecoderFactory>();
  config.task_queue_factory = CreateDefaultTaskQueueFactory();
  config.audio_processing_builder =
      std::make_unique<NiceMock<test::MockAudioProcessingBuilder>>();
  config.audio_device_module = test::MockAudioDeviceModule::CreateNice();

  auto voip_engine = CreateVoipEngine(std::move(config));
  EXPECT_NE(voip_engine, nullptr);
}

// Create voip engine without setting audio processing as optional component.
TEST(VoipEngineFactoryTest, UseNoAudioProcessing) {
  VoipEngineConfig config;
  config.encoder_factory = rtc::make_ref_counted<MockAudioEncoderFactory>();
  config.decoder_factory = rtc::make_ref_counted<MockAudioDecoderFactory>();
  config.task_queue_factory = CreateDefaultTaskQueueFactory();
  config.audio_device_module = test::MockAudioDeviceModule::CreateNice();

  auto voip_engine = CreateVoipEngine(std::move(config));
  EXPECT_NE(voip_engine, nullptr);
}

}  // namespace
}  // namespace webrtc
