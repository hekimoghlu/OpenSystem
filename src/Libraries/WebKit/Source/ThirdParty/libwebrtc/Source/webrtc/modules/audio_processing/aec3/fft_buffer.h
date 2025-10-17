/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_FFT_BUFFER_H_
#define MODULES_AUDIO_PROCESSING_AEC3_FFT_BUFFER_H_

#include <stddef.h>

#include <vector>

#include "modules/audio_processing/aec3/fft_data.h"
#include "rtc_base/checks.h"

namespace webrtc {

// Struct for bundling a circular buffer of FftData objects together with the
// read and write indices.
struct FftBuffer {
  FftBuffer(size_t size, size_t num_channels);
  ~FftBuffer();

  int IncIndex(int index) const {
    RTC_DCHECK_EQ(buffer.size(), static_cast<size_t>(size));
    return index < size - 1 ? index + 1 : 0;
  }

  int DecIndex(int index) const {
    RTC_DCHECK_EQ(buffer.size(), static_cast<size_t>(size));
    return index > 0 ? index - 1 : size - 1;
  }

  int OffsetIndex(int index, int offset) const {
    RTC_DCHECK_GE(buffer.size(), offset);
    RTC_DCHECK_EQ(buffer.size(), static_cast<size_t>(size));
    return (size + index + offset) % size;
  }

  void UpdateWriteIndex(int offset) { write = OffsetIndex(write, offset); }
  void IncWriteIndex() { write = IncIndex(write); }
  void DecWriteIndex() { write = DecIndex(write); }
  void UpdateReadIndex(int offset) { read = OffsetIndex(read, offset); }
  void IncReadIndex() { read = IncIndex(read); }
  void DecReadIndex() { read = DecIndex(read); }

  const int size;
  std::vector<std::vector<FftData>> buffer;
  int write = 0;
  int read = 0;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_FFT_BUFFER_H_
