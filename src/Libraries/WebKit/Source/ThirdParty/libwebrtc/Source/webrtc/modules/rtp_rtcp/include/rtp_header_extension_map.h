/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
#ifndef MODULES_RTP_RTCP_INCLUDE_RTP_HEADER_EXTENSION_MAP_H_
#define MODULES_RTP_RTCP_INCLUDE_RTP_HEADER_EXTENSION_MAP_H_

#include <stdint.h>

#include <string>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/rtp_parameters.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "rtc_base/checks.h"

namespace webrtc {

class RtpHeaderExtensionMap {
 public:
  static constexpr RTPExtensionType kInvalidType = kRtpExtensionNone;
  static constexpr int kInvalidId = 0;

  RtpHeaderExtensionMap();
  explicit RtpHeaderExtensionMap(bool extmap_allow_mixed);
  explicit RtpHeaderExtensionMap(rtc::ArrayView<const RtpExtension> extensions);

  void Reset(rtc::ArrayView<const RtpExtension> extensions);

  template <typename Extension>
  bool Register(int id) {
    return Register(id, Extension::kId, Extension::Uri());
  }
  bool RegisterByType(int id, RTPExtensionType type);
  bool RegisterByUri(int id, absl::string_view uri);

  bool IsRegistered(RTPExtensionType type) const {
    return GetId(type) != kInvalidId;
  }
  // Return kInvalidType if not found.
  RTPExtensionType GetType(int id) const;
  // Return kInvalidId if not found.
  uint8_t GetId(RTPExtensionType type) const {
    RTC_DCHECK_GT(type, kRtpExtensionNone);
    RTC_DCHECK_LT(type, kRtpExtensionNumberOfExtensions);
    return ids_[type];
  }

  void Deregister(absl::string_view uri);

  // Corresponds to the SDP attribute extmap-allow-mixed, see RFC8285.
  // Set to true if it's allowed to mix one- and two-byte RTP header extensions
  // in the same stream.
  bool ExtmapAllowMixed() const { return extmap_allow_mixed_; }
  void SetExtmapAllowMixed(bool extmap_allow_mixed) {
    extmap_allow_mixed_ = extmap_allow_mixed;
  }

 private:
  bool Register(int id, RTPExtensionType type, absl::string_view uri);

  uint8_t ids_[kRtpExtensionNumberOfExtensions];
  bool extmap_allow_mixed_;
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_INCLUDE_RTP_HEADER_EXTENSION_MAP_H_
