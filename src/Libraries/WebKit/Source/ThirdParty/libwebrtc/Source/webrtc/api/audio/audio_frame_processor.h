/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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
#ifndef API_AUDIO_AUDIO_FRAME_PROCESSOR_H_
#define API_AUDIO_AUDIO_FRAME_PROCESSOR_H_

#include <functional>
#include <memory>

namespace webrtc {

class AudioFrame;

// If passed into PeerConnectionFactory, will be used for additional
// processing of captured audio frames, performed before encoding.
// Implementations must be thread-safe.
class AudioFrameProcessor {
 public:
  using OnAudioFrameCallback = std::function<void(std::unique_ptr<AudioFrame>)>;
  virtual ~AudioFrameProcessor() = default;

  // Processes the frame received from WebRTC, is called by WebRTC off the
  // realtime audio capturing path. AudioFrameProcessor must reply with
  // processed frames by calling `sink_callback` if it was provided in SetSink()
  // call. `sink_callback` can be called in the context of Process().
  virtual void Process(std::unique_ptr<AudioFrame> frame) = 0;

  // Atomically replaces the current sink with the new one. Before the
  // first call to this function, or if the provided `sink_callback` is nullptr,
  // processed frames are simply discarded.
  virtual void SetSink(OnAudioFrameCallback sink_callback) = 0;
};

}  // namespace webrtc

#endif  // API_AUDIO_AUDIO_FRAME_PROCESSOR_H_
