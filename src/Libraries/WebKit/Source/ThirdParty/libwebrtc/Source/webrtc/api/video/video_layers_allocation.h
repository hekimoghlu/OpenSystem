/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 29, 2024.
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
#ifndef API_VIDEO_VIDEO_LAYERS_ALLOCATION_H_
#define API_VIDEO_VIDEO_LAYERS_ALLOCATION_H_

#include <cstdint>

#include "absl/container/inlined_vector.h"
#include "api/units/data_rate.h"

namespace webrtc {

// This struct contains additional stream-level information needed by a
// Selective Forwarding Middlebox to make relay decisions of RTP streams.
struct VideoLayersAllocation {
  static constexpr int kMaxSpatialIds = 4;
  static constexpr int kMaxTemporalIds = 4;

  friend bool operator==(const VideoLayersAllocation& lhs,
                         const VideoLayersAllocation& rhs) {
    return lhs.rtp_stream_index == rhs.rtp_stream_index &&
           lhs.resolution_and_frame_rate_is_valid ==
               rhs.resolution_and_frame_rate_is_valid &&
           lhs.active_spatial_layers == rhs.active_spatial_layers;
  }

  friend bool operator!=(const VideoLayersAllocation& lhs,
                         const VideoLayersAllocation& rhs) {
    return !(lhs == rhs);
  }

  struct SpatialLayer {
    friend bool operator==(const SpatialLayer& lhs, const SpatialLayer& rhs) {
      return lhs.rtp_stream_index == rhs.rtp_stream_index &&
             lhs.spatial_id == rhs.spatial_id &&
             lhs.target_bitrate_per_temporal_layer ==
                 rhs.target_bitrate_per_temporal_layer &&
             lhs.width == rhs.width && lhs.height == rhs.height &&
             lhs.frame_rate_fps == rhs.frame_rate_fps;
    }

    friend bool operator!=(const SpatialLayer& lhs, const SpatialLayer& rhs) {
      return !(lhs == rhs);
    }
    int rtp_stream_index = 0;
    // Index of the spatial layer per `rtp_stream_index`.
    int spatial_id = 0;
    // Target bitrate per decode target.
    absl::InlinedVector<DataRate, kMaxTemporalIds>
        target_bitrate_per_temporal_layer;

    // These fields are only valid if `resolution_and_frame_rate_is_valid` is
    // true
    uint16_t width = 0;
    uint16_t height = 0;
    // Max frame rate used in any temporal layer of this spatial layer.
    uint8_t frame_rate_fps = 0;
  };

  // Index of the rtp stream this allocation is sent on. Used for mapping
  // a SpatialLayer to a rtp stream.
  int rtp_stream_index = 0;
  bool resolution_and_frame_rate_is_valid = false;
  absl::InlinedVector<SpatialLayer, kMaxSpatialIds> active_spatial_layers;
};

}  // namespace webrtc

#endif  // API_VIDEO_VIDEO_LAYERS_ALLOCATION_H_
