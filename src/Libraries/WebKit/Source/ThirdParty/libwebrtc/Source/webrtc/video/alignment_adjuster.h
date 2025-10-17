/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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
#ifndef VIDEO_ALIGNMENT_ADJUSTER_H_
#define VIDEO_ALIGNMENT_ADJUSTER_H_

#include "api/video_codecs/video_encoder.h"
#include "video/config/video_encoder_config.h"

namespace webrtc {

class AlignmentAdjuster {
 public:
  // Returns the resolution alignment requested by the encoder (i.e
  // `EncoderInfo::requested_resolution_alignment` which ensures that delivered
  // frames to the encoder are divisible by this alignment).
  //
  // If `EncoderInfo::apply_alignment_to_all_simulcast_layers` is enabled, the
  // alignment will be adjusted to ensure that each simulcast layer also is
  // divisible by `requested_resolution_alignment`. The configured scale factors
  // `scale_resolution_down_by` may be adjusted to a common multiple to limit
  // the alignment value to avoid largely cropped frames and possibly with an
  // aspect ratio far from the original.

  // Note: `max_layers` currently only taken into account when using default
  // scale factors.
  static int GetAlignmentAndMaybeAdjustScaleFactors(
      const VideoEncoder::EncoderInfo& info,
      VideoEncoderConfig* config,
      std::optional<size_t> max_layers);
};

}  // namespace webrtc

#endif  // VIDEO_ALIGNMENT_ADJUSTER_H_
