/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 26, 2024.
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
#ifndef PC_RTP_MEDIA_UTILS_H_
#define PC_RTP_MEDIA_UTILS_H_

#include "api/rtp_transceiver_direction.h"

namespace webrtc {

// Returns the RtpTransceiverDirection that satisfies specified send and receive
// conditions.
RtpTransceiverDirection RtpTransceiverDirectionFromSendRecv(bool send,
                                                            bool recv);

// Returns true only if the direction will send media.
bool RtpTransceiverDirectionHasSend(RtpTransceiverDirection direction);

// Returns true only if the direction will receive media.
bool RtpTransceiverDirectionHasRecv(RtpTransceiverDirection direction);

// Returns the RtpTransceiverDirection which is the reverse of the given
// direction.
RtpTransceiverDirection RtpTransceiverDirectionReversed(
    RtpTransceiverDirection direction);

// Returns the RtpTransceiverDirection with its send component set to `send`.
RtpTransceiverDirection RtpTransceiverDirectionWithSendSet(
    RtpTransceiverDirection direction,
    bool send = true);

// Returns the RtpTransceiverDirection with its recv component set to `recv`.
RtpTransceiverDirection RtpTransceiverDirectionWithRecvSet(
    RtpTransceiverDirection direction,
    bool recv = true);

// Returns an unspecified string representation of the given direction.
const char* RtpTransceiverDirectionToString(RtpTransceiverDirection direction);

// Returns the intersection of the directions of two transceivers.
RtpTransceiverDirection RtpTransceiverDirectionIntersection(
    RtpTransceiverDirection lhs,
    RtpTransceiverDirection rhs);

}  // namespace webrtc

#endif  // PC_RTP_MEDIA_UTILS_H_
