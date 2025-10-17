/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 28, 2024.
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
#ifndef MEDIA_BASE_AUDIO_SOURCE_H_
#define MEDIA_BASE_AUDIO_SOURCE_H_

#include <cstddef>
#include <optional>

namespace cricket {

// Abstract interface for providing the audio data.
// TODO(deadbeef): Rename this to AudioSourceInterface, and rename
// webrtc::AudioSourceInterface to AudioTrackSourceInterface.
class AudioSource {
 public:
  class Sink {
   public:
    // Callback to receive data from the AudioSource.
    virtual void OnData(
        const void* audio_data,
        int bits_per_sample,
        int sample_rate,
        size_t number_of_channels,
        size_t number_of_frames,
        std::optional<int64_t> absolute_capture_timestamp_ms) = 0;

    // Called when the AudioSource is going away.
    virtual void OnClose() = 0;

    // Returns the number of channels encoded by the sink. This can be less than
    // the number_of_channels if down-mixing occur. A value of -1 means an
    // unknown number.
    virtual int NumPreferredChannels() const = 0;

   protected:
    virtual ~Sink() {}
  };

  // Sets a sink to the AudioSource. There can be only one sink connected
  // to the source at a time.
  virtual void SetSink(Sink* sink) = 0;

 protected:
  virtual ~AudioSource() {}
};

}  // namespace cricket

#endif  // MEDIA_BASE_AUDIO_SOURCE_H_
