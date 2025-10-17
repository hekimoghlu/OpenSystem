/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 30, 2022.
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
#include "modules/audio_processing/aec3/downsampled_render_buffer.h"

#include <algorithm>

namespace webrtc {

DownsampledRenderBuffer::DownsampledRenderBuffer(size_t downsampled_buffer_size)
    : size(static_cast<int>(downsampled_buffer_size)),
      buffer(downsampled_buffer_size, 0.f) {
  std::fill(buffer.begin(), buffer.end(), 0.f);
}

DownsampledRenderBuffer::~DownsampledRenderBuffer() = default;

}  // namespace webrtc
