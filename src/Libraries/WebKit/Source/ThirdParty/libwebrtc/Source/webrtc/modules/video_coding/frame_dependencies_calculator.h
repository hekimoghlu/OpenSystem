/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 28, 2022.
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
#ifndef MODULES_VIDEO_CODING_FRAME_DEPENDENCIES_CALCULATOR_H_
#define MODULES_VIDEO_CODING_FRAME_DEPENDENCIES_CALCULATOR_H_

#include <stdint.h>

#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "api/array_view.h"
#include "common_video/generic_frame_descriptor/generic_frame_info.h"

namespace webrtc {

// This class is thread compatible.
class FrameDependenciesCalculator {
 public:
  FrameDependenciesCalculator() = default;
  FrameDependenciesCalculator(const FrameDependenciesCalculator&) = default;
  FrameDependenciesCalculator& operator=(const FrameDependenciesCalculator&) =
      default;

  // Calculates frame dependencies based on previous encoder buffer usage.
  absl::InlinedVector<int64_t, 5> FromBuffersUsage(
      int64_t frame_id,
      rtc::ArrayView<const CodecBufferUsage> buffers_usage);

 private:
  struct BufferUsage {
    std::optional<int64_t> frame_id;
    absl::InlinedVector<int64_t, 4> dependencies;
  };

  absl::InlinedVector<BufferUsage, 4> buffers_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_FRAME_DEPENDENCIES_CALCULATOR_H_
