/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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
#ifndef MODULES_AUDIO_DEVICE_MOCK_AUDIO_DEVICE_BUFFER_H_
#define MODULES_AUDIO_DEVICE_MOCK_AUDIO_DEVICE_BUFFER_H_

#include <optional>

#include "modules/audio_device/audio_device_buffer.h"
#include "test/gmock.h"

namespace webrtc {

class MockAudioDeviceBuffer : public AudioDeviceBuffer {
 public:
  using AudioDeviceBuffer::AudioDeviceBuffer;
  virtual ~MockAudioDeviceBuffer() {}
  MOCK_METHOD(int32_t, RequestPlayoutData, (size_t nSamples), (override));
  MOCK_METHOD(int32_t, GetPlayoutData, (void* audioBuffer), (override));
  MOCK_METHOD(int32_t,
              SetRecordedBuffer,
              (const void* audioBuffer,
               size_t nSamples,
               std::optional<int64_t> capture_time_ns),
              (override));
  MOCK_METHOD(void, SetVQEData, (int playDelayMS, int recDelayMS), (override));
  MOCK_METHOD(int32_t, DeliverRecordedData, (), (override));
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_DEVICE_MOCK_AUDIO_DEVICE_BUFFER_H_
