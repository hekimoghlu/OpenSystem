/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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
#ifndef COMMON_VIDEO_H265_H265_PPS_PARSER_H_
#define COMMON_VIDEO_H265_H265_PPS_PARSER_H_

#include <optional>

#include "api/array_view.h"
#include "common_video/h265/h265_sps_parser.h"
#include "rtc_base/bitstream_reader.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// A class for parsing out picture parameter set (PPS) data from a H265 NALU.
class RTC_EXPORT H265PpsParser {
 public:
  // The parsed state of the PPS. Only some select values are stored.
  // Add more as they are actually needed.
  struct PpsState {
    PpsState() = default;

    bool dependent_slice_segments_enabled_flag = false;
    bool cabac_init_present_flag = false;
    bool output_flag_present_flag = false;
    uint32_t num_extra_slice_header_bits = 0;
    uint32_t num_ref_idx_l0_default_active_minus1 = 0;
    uint32_t num_ref_idx_l1_default_active_minus1 = 0;
    int init_qp_minus26 = 0;
    bool weighted_pred_flag = false;
    bool weighted_bipred_flag = false;
    bool lists_modification_present_flag = false;
    uint32_t pps_id = 0;
    uint32_t sps_id = 0;
    int qp_bd_offset_y = 0;
  };

  // Unpack RBSP and parse PPS state from the supplied buffer.
  static std::optional<PpsState> ParsePps(rtc::ArrayView<const uint8_t> data,
                                          const H265SpsParser::SpsState* sps);
  // TODO: bugs.webrtc.org/42225170 - Deprecate.
  static inline std::optional<PpsState> ParsePps(
      const uint8_t* data,
      size_t length,
      const H265SpsParser::SpsState* sps) {
    return ParsePps(rtc::MakeArrayView(data, length), sps);
  }

  static bool ParsePpsIds(rtc::ArrayView<const uint8_t> data,
                          uint32_t* pps_id,
                          uint32_t* sps_id);
  // TODO: bugs.webrtc.org/42225170 - Deprecate.
  static inline bool ParsePpsIds(const uint8_t* data,
                                 size_t length,
                                 uint32_t* pps_id,
                                 uint32_t* sps_id) {
    return ParsePpsIds(rtc::MakeArrayView(data, length), pps_id, sps_id);
  }

 protected:
  // Parse the PPS state, for a bit buffer where RBSP decoding has already been
  // performed.
  static std::optional<PpsState> ParseInternal(
      rtc::ArrayView<const uint8_t> buffer,
      const H265SpsParser::SpsState* sps);
  static bool ParsePpsIdsInternal(BitstreamReader& reader,
                                  uint32_t& pps_id,
                                  uint32_t& sps_id);
};

}  // namespace webrtc

#endif  // COMMON_VIDEO_H265_H265_PPS_PARSER_H_
