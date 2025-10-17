/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 21, 2024.
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
#include "api/rtc_error.h"

#include <iterator>
#include <string>

#include "absl/strings/string_view.h"

namespace {

absl::string_view kRTCErrorTypeNames[] = {
    "NONE",
    "UNSUPPORTED_OPERATION",
    "UNSUPPORTED_PARAMETER",
    "INVALID_PARAMETER",
    "INVALID_RANGE",
    "SYNTAX_ERROR",
    "INVALID_STATE",
    "INVALID_MODIFICATION",
    "NETWORK_ERROR",
    "RESOURCE_EXHAUSTED",
    "INTERNAL_ERROR",
    "OPERATION_ERROR_WITH_DATA",
};
static_assert(
    static_cast<int>(webrtc::RTCErrorType::OPERATION_ERROR_WITH_DATA) ==
        (std::size(kRTCErrorTypeNames) - 1),
    "kRTCErrorTypeNames must have as many strings as RTCErrorType "
    "has values.");

absl::string_view kRTCErrorDetailTypeNames[] = {
    "NONE",
    "DATA_CHANNEL_FAILURE",
    "DTLS_FAILURE",
    "FINGERPRINT_FAILURE",
    "SCTP_FAILURE",
    "SDP_SYNTAX_ERROR",
    "HARDWARE_ENCODER_NOT_AVAILABLE",
    "HARDWARE_ENCODER_ERROR",
};
static_assert(
    static_cast<int>(webrtc::RTCErrorDetailType::HARDWARE_ENCODER_ERROR) ==
        (std::size(kRTCErrorDetailTypeNames) - 1),
    "kRTCErrorDetailTypeNames must have as many strings as "
    "RTCErrorDetailType has values.");

}  // namespace

namespace webrtc {

// static
RTCError RTCError::OK() {
  return RTCError();
}

const char* RTCError::message() const {
  return message_.c_str();
}

void RTCError::set_message(absl::string_view message) {
  message_ = std::string(message);
}

absl::string_view ToString(RTCErrorType error) {
  int index = static_cast<int>(error);
  return kRTCErrorTypeNames[index];
}

absl::string_view ToString(RTCErrorDetailType error) {
  int index = static_cast<int>(error);
  return kRTCErrorDetailTypeNames[index];
}

}  // namespace webrtc
