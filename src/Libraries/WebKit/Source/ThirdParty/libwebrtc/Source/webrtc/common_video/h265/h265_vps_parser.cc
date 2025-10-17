/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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
#include "common_video/h265/h265_vps_parser.h"

#include "common_video/h265/h265_common.h"
#include "rtc_base/bit_buffer.h"
#include "rtc_base/bitstream_reader.h"
#include "rtc_base/logging.h"

#if WEBRTC_WEBKIT_BUILD
#include "common_video/h265/h265_sps_parser.h"
#endif

namespace webrtc {

H265VpsParser::VpsState::VpsState() = default;

// General note: this is based off the 08/2021 version of the H.265 standard.
// You can find it on this page:
// http://www.itu.int/rec/T-REC-H.265

// Unpack RBSP and parse VPS state from the supplied buffer.
std::optional<H265VpsParser::VpsState> H265VpsParser::ParseVps(
    rtc::ArrayView<const uint8_t> data) {
  return ParseInternal(H265::ParseRbsp(data));
}

std::optional<H265VpsParser::VpsState> H265VpsParser::ParseInternal(
    rtc::ArrayView<const uint8_t> buffer) {
  BitstreamReader reader(buffer);

  // Now, we need to use a bit buffer to parse through the actual H265 VPS
  // format. See Section 7.3.2.1 ("Video parameter set RBSP syntax") of the
  // H.265 standard for a complete description.
  VpsState vps;

  // vps_video_parameter_set_id: u(4)
  vps.id = reader.ReadBits(4);

  if (!reader.Ok()) {
    return std::nullopt;
  }

#if WEBRTC_WEBKIT_BUILD
  // vps_base_layer_internal_flag u(1)
  reader.ConsumeBits(1);
  // vps_base_layer_available_flag u(1)
  reader.ConsumeBits(1);
  // vps_max_layers_minus1 u(6)
  vps.vps_max_sub_layers_minus1 = reader.ReadBits(6);

  if (!reader.Ok() || (vps.vps_max_sub_layers_minus1 >= kMaxSubLayers)) {
    return std::nullopt;
  }

  //  vps_max_sub_layers_minus1 u(3)
  reader.ConsumeBits(3);
  //  vps_temporal_id_nesting_flag u(1)
  reader.ConsumeBits(1);
  //  vps_reserved_0xffff_16bits u(16)
  reader.ConsumeBits(16);

  auto profile_tier_level = H265SpsParser::ParseProfileTierLevel(true, vps.vps_max_sub_layers_minus1, reader);
  if (!reader.Ok() || !profile_tier_level) {
    return std::nullopt;
  }

  bool vps_sub_layer_ordering_info_present_flag = reader.Read<bool>();
  for (uint32_t i = (vps_sub_layer_ordering_info_present_flag != 0) ? 0 : vps.vps_max_sub_layers_minus1; i <= vps.vps_max_sub_layers_minus1; i++) {
    // vps_max_dec_pic_buffering_minus1[ i ]: ue(v)
    reader.ReadExponentialGolomb();
    // vps_max_num_reorder_pics[ i ]: ue(v)
    vps.vps_max_num_reorder_pics[i] = reader.ReadExponentialGolomb();
    if (!reader.Ok() || (i > 0 && vps.vps_max_num_reorder_pics[i] < vps.vps_max_num_reorder_pics[i - 1])) {
      return std::nullopt;
    }

    // vps_max_latency_increase_plus1: ue(v)
    reader.ReadExponentialGolomb();
  }
  if (!vps_sub_layer_ordering_info_present_flag) {
    for (int i = 0; i < vps.vps_max_sub_layers_minus1; ++i) {
      vps.vps_max_num_reorder_pics[i] = vps.vps_max_num_reorder_pics[vps.vps_max_sub_layers_minus1];
    }
  }
  if (!reader.Ok() || !profile_tier_level) {
    return std::nullopt;
  }
#endif

  return vps;
}

}  // namespace webrtc
