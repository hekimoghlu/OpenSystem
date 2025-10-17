/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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
#ifndef API_CALL_AUDIO_SINK_H_
#define API_CALL_AUDIO_SINK_H_

#include <stddef.h>
#include <stdint.h>

namespace webrtc {

// Represents a simple push audio sink.
class AudioSinkInterface {
 public:
  virtual ~AudioSinkInterface() {}

  struct Data {
    Data(const int16_t* data,
         size_t samples_per_channel,
         int sample_rate,
         size_t channels,
         uint32_t timestamp)
        : data(data),
          samples_per_channel(samples_per_channel),
          sample_rate(sample_rate),
          channels(channels),
          timestamp(timestamp) {}

    const int16_t* data;         // The actual 16bit audio data.
    size_t samples_per_channel;  // Number of frames in the buffer.
    int sample_rate;             // Sample rate in Hz.
    size_t channels;             // Number of channels in the audio data.
    uint32_t timestamp;          // The RTP timestamp of the first sample.
  };

  virtual void OnData(const Data& audio) = 0;
};

}  // namespace webrtc

#endif  // API_CALL_AUDIO_SINK_H_
