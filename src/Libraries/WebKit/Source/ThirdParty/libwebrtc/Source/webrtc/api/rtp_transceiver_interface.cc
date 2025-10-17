/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 28, 2025.
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
#include "api/rtp_transceiver_interface.h"

#include <optional>

#include "api/rtc_error.h"
#include "api/rtp_transceiver_direction.h"
#include "rtc_base/checks.h"

namespace webrtc {

RtpTransceiverInit::RtpTransceiverInit() = default;

RtpTransceiverInit::RtpTransceiverInit(const RtpTransceiverInit& rhs) = default;

RtpTransceiverInit::~RtpTransceiverInit() = default;

std::optional<RtpTransceiverDirection>
RtpTransceiverInterface::fired_direction() const {
  return std::nullopt;
}

bool RtpTransceiverInterface::stopping() const {
  return false;
}

void RtpTransceiverInterface::Stop() {
  StopInternal();
}

RTCError RtpTransceiverInterface::StopStandard() {
  RTC_DCHECK_NOTREACHED()
      << "DEBUG: RtpTransceiverInterface::StopStandard called";
  return RTCError::OK();
}

void RtpTransceiverInterface::StopInternal() {
  RTC_DCHECK_NOTREACHED()
      << "DEBUG: RtpTransceiverInterface::StopInternal called";
}

// TODO(bugs.webrtc.org/11839) Remove default implementations when clients
// are updated.
void RtpTransceiverInterface::SetDirection(
    RtpTransceiverDirection new_direction) {
  SetDirectionWithError(new_direction);
}

RTCError RtpTransceiverInterface::SetDirectionWithError(
    RtpTransceiverDirection /* new_direction */) {
  RTC_DCHECK_NOTREACHED() << "Default implementation called";
  return RTCError::OK();
}

}  // namespace webrtc
