/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 12, 2022.
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
#include "pc/rtp_media_utils.h"

#include "rtc_base/checks.h"

namespace webrtc {

RtpTransceiverDirection RtpTransceiverDirectionFromSendRecv(bool send,
                                                            bool recv) {
  if (send && recv) {
    return RtpTransceiverDirection::kSendRecv;
  } else if (send && !recv) {
    return RtpTransceiverDirection::kSendOnly;
  } else if (!send && recv) {
    return RtpTransceiverDirection::kRecvOnly;
  } else {
    return RtpTransceiverDirection::kInactive;
  }
}

bool RtpTransceiverDirectionHasSend(RtpTransceiverDirection direction) {
  return direction == RtpTransceiverDirection::kSendRecv ||
         direction == RtpTransceiverDirection::kSendOnly;
}

bool RtpTransceiverDirectionHasRecv(RtpTransceiverDirection direction) {
  return direction == RtpTransceiverDirection::kSendRecv ||
         direction == RtpTransceiverDirection::kRecvOnly;
}

RtpTransceiverDirection RtpTransceiverDirectionReversed(
    RtpTransceiverDirection direction) {
  switch (direction) {
    case RtpTransceiverDirection::kSendRecv:
    case RtpTransceiverDirection::kInactive:
    case RtpTransceiverDirection::kStopped:
      return direction;
    case RtpTransceiverDirection::kSendOnly:
      return RtpTransceiverDirection::kRecvOnly;
    case RtpTransceiverDirection::kRecvOnly:
      return RtpTransceiverDirection::kSendOnly;
    default:
      RTC_DCHECK_NOTREACHED();
      return direction;
  }
}

RtpTransceiverDirection RtpTransceiverDirectionWithSendSet(
    RtpTransceiverDirection direction,
    bool send) {
  return RtpTransceiverDirectionFromSendRecv(
      send, RtpTransceiverDirectionHasRecv(direction));
}

RtpTransceiverDirection RtpTransceiverDirectionWithRecvSet(
    RtpTransceiverDirection direction,
    bool recv) {
  return RtpTransceiverDirectionFromSendRecv(
      RtpTransceiverDirectionHasSend(direction), recv);
}

const char* RtpTransceiverDirectionToString(RtpTransceiverDirection direction) {
  switch (direction) {
    case RtpTransceiverDirection::kSendRecv:
      return "kSendRecv";
    case RtpTransceiverDirection::kSendOnly:
      return "kSendOnly";
    case RtpTransceiverDirection::kRecvOnly:
      return "kRecvOnly";
    case RtpTransceiverDirection::kInactive:
      return "kInactive";
    case RtpTransceiverDirection::kStopped:
      return "kStopped";
  }
  RTC_DCHECK_NOTREACHED();
  return "";
}

RtpTransceiverDirection RtpTransceiverDirectionIntersection(
    RtpTransceiverDirection lhs,
    RtpTransceiverDirection rhs) {
  return RtpTransceiverDirectionFromSendRecv(
      RtpTransceiverDirectionHasSend(lhs) &&
          RtpTransceiverDirectionHasSend(rhs),
      RtpTransceiverDirectionHasRecv(lhs) &&
          RtpTransceiverDirectionHasRecv(rhs));
}

}  // namespace webrtc
