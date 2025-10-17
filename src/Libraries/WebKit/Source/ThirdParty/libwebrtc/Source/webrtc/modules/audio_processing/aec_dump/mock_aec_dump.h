/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC_DUMP_MOCK_AEC_DUMP_H_
#define MODULES_AUDIO_PROCESSING_AEC_DUMP_MOCK_AEC_DUMP_H_

#include <memory>

#include "modules/audio_processing/include/aec_dump.h"
#include "test/gmock.h"

namespace webrtc {

namespace test {

class MockAecDump : public AecDump {
 public:
  MockAecDump();
  virtual ~MockAecDump();

  MOCK_METHOD(void,
              WriteInitMessage,
              (const ProcessingConfig& api_format, int64_t time_now_ms),
              (override));

  MOCK_METHOD(void,
              AddCaptureStreamInput,
              (const AudioFrameView<const float>& src),
              (override));
  MOCK_METHOD(void,
              AddCaptureStreamOutput,
              (const AudioFrameView<const float>& src),
              (override));
  MOCK_METHOD(void,
              AddCaptureStreamInput,
              (const int16_t* const data,
               int num_channels,
               int samples_per_channel),
              (override));
  MOCK_METHOD(void,
              AddCaptureStreamOutput,
              (const int16_t* const data,
               int num_channels,
               int samples_per_channel),
              (override));
  MOCK_METHOD(void,
              AddAudioProcessingState,
              (const AudioProcessingState& state),
              (override));
  MOCK_METHOD(void, WriteCaptureStreamMessage, (), (override));

  MOCK_METHOD(void,
              WriteRenderStreamMessage,
              (const int16_t* const data,
               int num_channels,
               int samples_per_channel),
              (override));
  MOCK_METHOD(void,
              WriteRenderStreamMessage,
              (const AudioFrameView<const float>& src),
              (override));

  MOCK_METHOD(void, WriteConfig, (const InternalAPMConfig& config), (override));

  MOCK_METHOD(void,
              WriteRuntimeSetting,
              (const AudioProcessing::RuntimeSetting& config),
              (override));
};

}  // namespace test

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC_DUMP_MOCK_AEC_DUMP_H_
