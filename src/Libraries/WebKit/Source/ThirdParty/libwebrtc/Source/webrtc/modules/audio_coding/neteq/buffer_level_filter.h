/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_BUFFER_LEVEL_FILTER_H_
#define MODULES_AUDIO_CODING_NETEQ_BUFFER_LEVEL_FILTER_H_

#include <stddef.h>
#include <stdint.h>

namespace webrtc {

class BufferLevelFilter {
 public:
  BufferLevelFilter();
  virtual ~BufferLevelFilter() {}

  BufferLevelFilter(const BufferLevelFilter&) = delete;
  BufferLevelFilter& operator=(const BufferLevelFilter&) = delete;

  virtual void Reset();

  // Updates the filter. Current buffer size is `buffer_size_samples`.
  // `time_stretched_samples` is subtracted from the filtered value (thus
  // bypassing the filter operation).
  virtual void Update(size_t buffer_size_samples, int time_stretched_samples);

  // Set the filtered buffer level to a particular value directly. This should
  // only be used in case of large changes in buffer size, such as buffer
  // flushes.
  virtual void SetFilteredBufferLevel(int buffer_size_samples);

  // The target level is used to select the appropriate filter coefficient.
  virtual void SetTargetBufferLevel(int target_buffer_level_ms);

  // Returns filtered current level in number of samples.
  virtual int filtered_current_level() const {
    // Round to nearest whole sample.
    return (int64_t{filtered_current_level_} + (1 << 7)) >> 8;
  }

 private:
  int level_factor_;  // Filter factor for the buffer level filter in Q8.
  int filtered_current_level_;  // Filtered current buffer level in Q8.
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_BUFFER_LEVEL_FILTER_H_
