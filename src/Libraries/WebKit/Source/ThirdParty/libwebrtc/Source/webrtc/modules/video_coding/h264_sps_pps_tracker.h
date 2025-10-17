/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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
#ifndef MODULES_VIDEO_CODING_H264_SPS_PPS_TRACKER_H_
#define MODULES_VIDEO_CODING_H264_SPS_PPS_TRACKER_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "api/array_view.h"
#include "modules/rtp_rtcp/source/rtp_video_header.h"
#include "rtc_base/buffer.h"
#include "rtc_base/copy_on_write_buffer.h"

namespace webrtc {
namespace video_coding {

class H264SpsPpsTracker {
 public:
  enum PacketAction { kInsert, kDrop, kRequestKeyframe };
  struct FixedBitstream {
    PacketAction action;
    rtc::CopyOnWriteBuffer bitstream;
  };

  H264SpsPpsTracker() = default;
  H264SpsPpsTracker(const H264SpsPpsTracker& other) = default;
  H264SpsPpsTracker& operator=(const H264SpsPpsTracker& other) = default;
  ~H264SpsPpsTracker() = default;

  // Returns fixed bitstream and modifies `video_header`.
  FixedBitstream CopyAndFixBitstream(rtc::ArrayView<const uint8_t> bitstream,
                                     RTPVideoHeader* video_header);

  void InsertSpsPpsNalus(const std::vector<uint8_t>& sps,
                         const std::vector<uint8_t>& pps);

 private:
  struct PpsInfo {
    int sps_id = -1;
    rtc::Buffer data;
  };

  struct SpsInfo {
    int width = -1;
    int height = -1;
    rtc::Buffer data;
  };

  std::map<int, PpsInfo> pps_data_;
  std::map<int, SpsInfo> sps_data_;
};

}  // namespace video_coding
}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_H264_SPS_PPS_TRACKER_H_
