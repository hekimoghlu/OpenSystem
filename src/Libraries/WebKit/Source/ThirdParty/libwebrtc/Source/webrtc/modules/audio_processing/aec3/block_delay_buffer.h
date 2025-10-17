/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_BLOCK_DELAY_BUFFER_H_
#define MODULES_AUDIO_PROCESSING_AEC3_BLOCK_DELAY_BUFFER_H_

#include <stddef.h>

#include <vector>

#include "modules/audio_processing/audio_buffer.h"

namespace webrtc {

// Class for applying a fixed delay to the samples in a signal partitioned using
// the audiobuffer band-splitting scheme.
class BlockDelayBuffer {
 public:
  BlockDelayBuffer(size_t num_channels,
                   size_t num_bands,
                   size_t frame_length,
                   size_t delay_samples);
  ~BlockDelayBuffer();

  // Delays the samples by the specified delay.
  void DelaySignal(AudioBuffer* frame);

 private:
  const size_t frame_length_;
  const size_t delay_;
  std::vector<std::vector<std::vector<float>>> buf_;
  size_t last_insert_ = 0;
};
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_BLOCK_DELAY_BUFFER_H_
