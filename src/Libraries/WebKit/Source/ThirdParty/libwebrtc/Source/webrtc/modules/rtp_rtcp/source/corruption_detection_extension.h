/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_CORRUPTION_DETECTION_EXTENSION_H_
#define MODULES_RTP_RTCP_SOURCE_CORRUPTION_DETECTION_EXTENSION_H_

#include <cstddef>
#include <cstdint>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/rtp_parameters.h"
#include "common_video/corruption_detection_message.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"

namespace webrtc {

// RTP Corruption Detection Header Extension.
//
// The class reads and writes the corruption detection RTP header extension.
// The class implements traits so that the class is compatible with being an
// argument to the templated `RtpPacket::GetExtension` and
// `RtpPacketToSend::SetExtension` methods.
class CorruptionDetectionExtension {
 public:
  using value_type = CorruptionDetectionMessage;

  static constexpr RTPExtensionType kId = kRtpExtensionCorruptionDetection;
  static constexpr uint8_t kMaxValueSizeBytes = 16;

  static constexpr absl::string_view Uri() {
    return RtpExtension::kCorruptionDetectionUri;
  }
  static bool Parse(rtc::ArrayView<const uint8_t> data,
                    CorruptionDetectionMessage* message);
  static bool Write(rtc::ArrayView<uint8_t> data,
                    const CorruptionDetectionMessage& message);
  // Size of the header extension in bytes.
  static size_t ValueSize(const CorruptionDetectionMessage& message);
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_CORRUPTION_DETECTION_EXTENSION_H_
