/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 25, 2025.
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
#ifndef MODULES_AUDIO_DEVICE_INCLUDE_MOCK_AUDIO_TRANSPORT_H_
#define MODULES_AUDIO_DEVICE_INCLUDE_MOCK_AUDIO_TRANSPORT_H_

#include "api/audio/audio_device_defines.h"
#include "test/gmock.h"

namespace webrtc {
namespace test {

class MockAudioTransport : public AudioTransport {
 public:
  MockAudioTransport() {}
  ~MockAudioTransport() {}

  MOCK_METHOD(int32_t,
              RecordedDataIsAvailable,
              (const void* audioSamples,
               size_t nSamples,
               size_t nBytesPerSample,
               size_t nChannels,
               uint32_t samplesPerSec,
               uint32_t totalDelayMS,
               int32_t clockDrift,
               uint32_t currentMicLevel,
               bool keyPressed,
               uint32_t& newMicLevel),
              (override));

  MOCK_METHOD(int32_t,
              RecordedDataIsAvailable,
              (const void* audioSamples,
               size_t nSamples,
               size_t nBytesPerSample,
               size_t nChannels,
               uint32_t samplesPerSec,
               uint32_t totalDelayMS,
               int32_t clockDrift,
               uint32_t currentMicLevel,
               bool keyPressed,
               uint32_t& newMicLevel,
               std::optional<int64_t> estimated_capture_time_ns),
              (override));

  MOCK_METHOD(int32_t,
              NeedMorePlayData,
              (size_t nSamples,
               size_t nBytesPerSample,
               size_t nChannels,
               uint32_t samplesPerSec,
               void* audioSamples,
               size_t& nSamplesOut,
               int64_t* elapsed_time_ms,
               int64_t* ntp_time_ms),
              (override));

  MOCK_METHOD(void,
              PullRenderData,
              (int bits_per_sample,
               int sample_rate,
               size_t number_of_channels,
               size_t number_of_frames,
               void* audio_data,
               int64_t* elapsed_time_ms,
               int64_t* ntp_time_ms),
              (override));
};

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_DEVICE_INCLUDE_MOCK_AUDIO_TRANSPORT_H_
