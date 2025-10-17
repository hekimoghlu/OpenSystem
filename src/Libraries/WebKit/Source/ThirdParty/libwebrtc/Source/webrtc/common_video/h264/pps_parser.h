/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 15, 2024.
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
#ifndef COMMON_VIDEO_H264_PPS_PARSER_H_
#define COMMON_VIDEO_H264_PPS_PARSER_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>

#include "api/array_view.h"

namespace webrtc {

// A class for parsing out picture parameter set (PPS) data from a H264 NALU.
class PpsParser {
 public:
  // The parsed state of the PPS. Only some select values are stored.
  // Add more as they are actually needed.
  struct PpsState {
    PpsState() = default;

    bool bottom_field_pic_order_in_frame_present_flag = false;
    bool weighted_pred_flag = false;
    bool entropy_coding_mode_flag = false;
    uint32_t num_ref_idx_l0_default_active_minus1 = 0;
    uint32_t num_ref_idx_l1_default_active_minus1 = 0;
    uint32_t weighted_bipred_idc = false;
    uint32_t redundant_pic_cnt_present_flag = 0;
    int pic_init_qp_minus26 = 0;
    uint32_t id = 0;
    uint32_t sps_id = 0;
  };

  struct SliceHeader {
    SliceHeader() = default;

    uint32_t first_mb_in_slice = 0;
    uint32_t pic_parameter_set_id = 0;
  };

  // Unpack RBSP and parse PPS state from the supplied buffer.
  static std::optional<PpsState> ParsePps(rtc::ArrayView<const uint8_t> data);
  // TODO: bugs.webrtc.org/42225170 - Deprecate.
  static inline std::optional<PpsState> ParsePps(const uint8_t* data,
                                                 size_t length) {
    return ParsePps(rtc::MakeArrayView(data, length));
  }

  static bool ParsePpsIds(rtc::ArrayView<const uint8_t> data,
                          uint32_t* pps_id,
                          uint32_t* sps_id);

  static std::optional<SliceHeader> ParseSliceHeader(
      rtc::ArrayView<const uint8_t> data);

 protected:
  // Parse the PPS state, for a buffer where RBSP decoding has already been
  // performed.
  static std::optional<PpsState> ParseInternal(
      rtc::ArrayView<const uint8_t> buffer);
};

}  // namespace webrtc

#endif  // COMMON_VIDEO_H264_PPS_PARSER_H_
