/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef COMMON_VIDEO_H264_SPS_PARSER_H_
#define COMMON_VIDEO_H264_SPS_PARSER_H_

#include <optional>

#include "rtc_base/bitstream_reader.h"
#include "rtc_base/system/rtc_export.h"

#if defined(WEBRTC_WEBKIT_BUILD)
#include <cstdint>
#endif

namespace webrtc {

// A class for parsing out sequence parameter set (SPS) data from an H264 NALU.
class RTC_EXPORT SpsParser {
 public:
  // The parsed state of the SPS. Only some select values are stored.
  // Add more as they are actually needed.
  struct RTC_EXPORT SpsState {
    SpsState();
    SpsState(const SpsState&);
    ~SpsState();

#if WEBRTC_WEBKIT_BUILD
    uint32_t pic_width_in_mbs_minus1 = 0;
    uint32_t pic_height_in_map_units_minus1 = 0;
#endif
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t delta_pic_order_always_zero_flag = 0;
    uint32_t chroma_format_idc = 1;
    uint32_t separate_colour_plane_flag = 0;
    uint32_t frame_mbs_only_flag = 0;
    uint32_t log2_max_frame_num = 4;          // Smallest valid value.
    uint32_t log2_max_pic_order_cnt_lsb = 4;  // Smallest valid value.
    uint32_t pic_order_cnt_type = 0;
    uint32_t max_num_ref_frames = 0;
    uint32_t vui_params_present = 0;
    uint32_t id = 0;
  };

  // Unpack RBSP and parse SPS state from the supplied buffer.
  static std::optional<SpsState> ParseSps(rtc::ArrayView<const uint8_t> data);

 protected:
  // Parse the SPS state, up till the VUI part, for a buffer where RBSP
  // decoding has already been performed.
  static std::optional<SpsState> ParseSpsUpToVui(BitstreamReader& reader);
};

}  // namespace webrtc
#endif  // COMMON_VIDEO_H264_SPS_PARSER_H_
