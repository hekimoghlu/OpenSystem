/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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
#ifndef API_TEST_MOCK_AUDIO_SINK_H_
#define API_TEST_MOCK_AUDIO_SINK_H_

#include <cstddef>
#include <cstdint>
#include <optional>

#include "api/media_stream_interface.h"
#include "test/gmock.h"

namespace webrtc {

class MockAudioSink : public webrtc::AudioTrackSinkInterface {
 public:
  MOCK_METHOD(void,
              OnData,
              (const void* audio_data,
               int bits_per_sample,
               int sample_rate,
               size_t number_of_channels,
               size_t number_of_frames),
              (override));

  MOCK_METHOD(void,
              OnData,
              (const void* audio_data,
               int bits_per_sample,
               int sample_rate,
               size_t number_of_channels,
               size_t number_of_frames,
               std::optional<int64_t> absolute_capture_timestamp_ms),
              (override));
};

}  // namespace webrtc

#endif  // API_TEST_MOCK_AUDIO_SINK_H_
