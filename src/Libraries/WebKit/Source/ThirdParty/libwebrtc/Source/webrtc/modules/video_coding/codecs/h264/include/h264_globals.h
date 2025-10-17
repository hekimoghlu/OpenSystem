/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
// This file contains codec dependent definitions that are needed in
// order to compile the WebRTC codebase, even if this codec is not used.

#ifndef MODULES_VIDEO_CODING_CODECS_H264_INCLUDE_H264_GLOBALS_H_
#define MODULES_VIDEO_CODING_CODECS_H264_INCLUDE_H264_GLOBALS_H_

#include <algorithm>
#include <string>
#include <vector>

#include "modules/video_coding/codecs/interface/common_constants.h"
#include "rtc_base/checks.h"

namespace webrtc {

// The packetization types that we support: single, aggregated, and fragmented.
enum H264PacketizationTypes {
  kH264SingleNalu,  // This packet contains a single NAL unit.
  kH264StapA,       // This packet contains STAP-A (single time
                    // aggregation) packets. If this packet has an
                    // associated NAL unit type, it'll be for the
                    // first such aggregated packet.
  kH264FuA,         // This packet contains a FU-A (fragmentation
                    // unit) packet, meaning it is a part of a frame
                    // that was too large to fit into a single packet.
};

// Packetization modes are defined in RFC 6184 section 6
// Due to the structure containing this being initialized with zeroes
// in some places, and mode 1 being default, mode 1 needs to have the value
// zero. https://crbug.com/webrtc/6803
enum class H264PacketizationMode {
  NonInterleaved = 0,  // Mode 1 - STAP-A, FU-A is allowed
  SingleNalUnit        // Mode 0 - only single NALU allowed
};

// This function is declared inline because it is not clear which
// .cc file it should belong to.
// TODO(hta): Refactor. https://bugs.webrtc.org/6842
// TODO(jonasolsson): Use absl::string_view instead when that's available.
inline std::string ToString(H264PacketizationMode mode) {
  if (mode == H264PacketizationMode::NonInterleaved) {
    return "NonInterleaved";
  } else if (mode == H264PacketizationMode::SingleNalUnit) {
    return "SingleNalUnit";
  }
  RTC_DCHECK_NOTREACHED();
  return "";
}

struct NaluInfo {
  uint8_t type;
  int sps_id;
  int pps_id;

  friend bool operator==(const NaluInfo& lhs, const NaluInfo& rhs) {
    return lhs.type == rhs.type && lhs.sps_id == rhs.sps_id &&
           lhs.pps_id == rhs.pps_id;
  }

  friend bool operator!=(const NaluInfo& lhs, const NaluInfo& rhs) {
    return !(lhs == rhs);
  }
};

struct RTPVideoHeaderH264 {
  // The NAL unit type. If this is a header for a
  // fragmented packet, it's the NAL unit type of
  // the original data. If this is the header for an
  // aggregated packet, it's the NAL unit type of
  // the first NAL unit in the packet.
  uint8_t nalu_type;
  // The packetization type of this buffer - single, aggregated or fragmented.
  H264PacketizationTypes packetization_type;
  std::vector<NaluInfo> nalus;
  // The packetization mode of this transport. Packetization mode
  // determines which packetization types are allowed when packetizing.
  H264PacketizationMode packetization_mode;

  friend bool operator==(const RTPVideoHeaderH264& lhs,
                         const RTPVideoHeaderH264& rhs) {
    return lhs.nalu_type == rhs.nalu_type &&
           lhs.packetization_type == rhs.packetization_type &&
           lhs.nalus == rhs.nalus &&
           lhs.packetization_mode == rhs.packetization_mode;
  }

  friend bool operator!=(const RTPVideoHeaderH264& lhs,
                         const RTPVideoHeaderH264& rhs) {
    return !(lhs == rhs);
  }
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_CODECS_H264_INCLUDE_H264_GLOBALS_H_
