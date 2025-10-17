/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
#include "modules/audio_processing/aec3/spectrum_buffer.h"

#include <algorithm>

namespace webrtc {

SpectrumBuffer::SpectrumBuffer(size_t size, size_t num_channels)
    : size(static_cast<int>(size)),
      buffer(size,
             std::vector<std::array<float, kFftLengthBy2Plus1>>(num_channels)) {
  for (auto& channel : buffer) {
    for (auto& c : channel) {
      std::fill(c.begin(), c.end(), 0.f);
    }
  }
}

SpectrumBuffer::~SpectrumBuffer() = default;

}  // namespace webrtc
