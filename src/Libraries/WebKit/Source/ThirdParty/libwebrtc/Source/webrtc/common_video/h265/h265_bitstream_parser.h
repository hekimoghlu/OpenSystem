/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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
#ifndef COMMON_VIDEO_H265_H265_BITSTREAM_PARSER_H_
#define COMMON_VIDEO_H265_H265_BITSTREAM_PARSER_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>
#include <vector>

#include "api/video_codecs/bitstream_parser.h"
#include "common_video/h265/h265_pps_parser.h"
#include "common_video/h265/h265_sps_parser.h"
#include "common_video/h265/h265_vps_parser.h"
#include "rtc_base/containers/flat_map.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Stateful H265 bitstream parser (due to VPS/SPS/PPS). Used to parse out QP
// values from the bitstream.
class RTC_EXPORT H265BitstreamParser : public BitstreamParser {
 public:
  H265BitstreamParser();
  ~H265BitstreamParser() override;

  // New interface.
  void ParseBitstream(rtc::ArrayView<const uint8_t> bitstream) override;
  std::optional<int> GetLastSliceQp() const override;

  std::optional<uint32_t> GetLastSlicePpsId() const;

  static std::optional<uint32_t> ParsePpsIdFromSliceSegmentLayerRbsp(
      rtc::ArrayView<const uint8_t> data,
      uint8_t nalu_type);

 protected:
  enum Result {
    kOk,
    kInvalidStream,
    kUnsupportedStream,
  };
  void ParseSlice(rtc::ArrayView<const uint8_t> slice);
  Result ParseNonParameterSetNalu(rtc::ArrayView<const uint8_t> source,
                                  uint8_t nalu_type);

  const H265PpsParser::PpsState* GetPPS(uint32_t id) const;
  const H265SpsParser::SpsState* GetSPS(uint32_t id) const;

  // VPS/SPS/PPS state, updated when parsing new VPS/SPS/PPS, used to parse
  // slices.
  flat_map<uint32_t, H265VpsParser::VpsState> vps_;
  flat_map<uint32_t, H265SpsParser::SpsState> sps_;
  flat_map<uint32_t, H265PpsParser::PpsState> pps_;

  // Last parsed slice QP.
  std::optional<int32_t> last_slice_qp_delta_;
  std::optional<uint32_t> last_slice_pps_id_;
};

}  // namespace webrtc

#endif  // COMMON_VIDEO_H265_H265_BITSTREAM_PARSER_H_
