/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 27, 2023.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_AUDIO_SINK_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_AUDIO_SINK_H_

#include "api/audio/audio_frame.h"

namespace webrtc {
namespace test {

// Interface class for an object receiving raw output audio from test
// applications.
class AudioSink {
 public:
  AudioSink() {}
  virtual ~AudioSink() {}

  AudioSink(const AudioSink&) = delete;
  AudioSink& operator=(const AudioSink&) = delete;

  // Writes `num_samples` from `audio` to the AudioSink. Returns true if
  // successful, otherwise false.
  virtual bool WriteArray(const int16_t* audio, size_t num_samples) = 0;

  // Writes `audio_frame` to the AudioSink. Returns true if successful,
  // otherwise false.
  bool WriteAudioFrame(const AudioFrame& audio_frame) {
    return WriteArray(audio_frame.data(), audio_frame.samples_per_channel_ *
                                              audio_frame.num_channels_);
  }
};

// Forks the output audio to two AudioSink objects.
class AudioSinkFork : public AudioSink {
 public:
  AudioSinkFork(AudioSink* left, AudioSink* right)
      : left_sink_(left), right_sink_(right) {}

  AudioSinkFork(const AudioSinkFork&) = delete;
  AudioSinkFork& operator=(const AudioSinkFork&) = delete;

  bool WriteArray(const int16_t* audio, size_t num_samples) override;

 private:
  AudioSink* left_sink_;
  AudioSink* right_sink_;
};

// An AudioSink implementation that does nothing.
class VoidAudioSink : public AudioSink {
 public:
  VoidAudioSink() = default;

  VoidAudioSink(const VoidAudioSink&) = delete;
  VoidAudioSink& operator=(const VoidAudioSink&) = delete;

  bool WriteArray(const int16_t* audio, size_t num_samples) override;
};

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_AUDIO_SINK_H_
