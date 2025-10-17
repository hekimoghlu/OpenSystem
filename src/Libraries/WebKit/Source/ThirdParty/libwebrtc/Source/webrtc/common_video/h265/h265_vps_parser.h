/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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
#ifndef COMMON_VIDEO_H265_H265_VPS_PARSER_H_
#define COMMON_VIDEO_H265_H265_VPS_PARSER_H_

#include <optional>

#include "api/array_view.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// A class for parsing out video parameter set (VPS) data from an H265 NALU.
class RTC_EXPORT H265VpsParser {
 public:
#if WEBRTC_WEBKIT_BUILD
    static constexpr uint32_t kMaxSubLayers = 7;
#endif

  // The parsed state of the VPS. Only some select values are stored.
  // Add more as they are actually needed.
  struct RTC_EXPORT VpsState {
    VpsState();

    uint32_t id = 0;
#if WEBRTC_WEBKIT_BUILD
    uint32_t vps_max_sub_layers_minus1 = 0;
    uint32_t vps_max_num_reorder_pics[kMaxSubLayers] = {};
#endif
  };

  // Unpack RBSP and parse VPS state from the supplied buffer.
  static std::optional<VpsState> ParseVps(rtc::ArrayView<const uint8_t> data);
  // TODO: bugs.webrtc.org/42225170 - Deprecate.
  static inline std::optional<VpsState> ParseVps(const uint8_t* data,
                                                 size_t length) {
    return ParseVps(rtc::MakeArrayView(data, length));
  }

 protected:
  // Parse the VPS state, for a bit buffer where RBSP decoding has already been
  // performed.
  static std::optional<VpsState> ParseInternal(
      rtc::ArrayView<const uint8_t> buffer);
};

}  // namespace webrtc
#endif  // COMMON_VIDEO_H265_H265_VPS_PARSER_H_
