/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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
#ifndef NET_DCSCTP_COMMON_INTERNAL_TYPES_H_
#define NET_DCSCTP_COMMON_INTERNAL_TYPES_H_

#include <functional>
#include <utility>

#include "net/dcsctp/public/types.h"
#include "rtc_base/strong_alias.h"

namespace dcsctp {

// Stream Sequence Number (SSN)
using SSN = webrtc::StrongAlias<class SSNTag, uint16_t>;

// Message Identifier (MID)
using MID = webrtc::StrongAlias<class MIDTag, uint32_t>;

// Fragment Sequence Number (FSN)
using FSN = webrtc::StrongAlias<class FSNTag, uint32_t>;

// Transmission Sequence Number (TSN)
using TSN = webrtc::StrongAlias<class TSNTag, uint32_t>;

// Reconfiguration Request Sequence Number
using ReconfigRequestSN =
    webrtc::StrongAlias<class ReconfigRequestSNTag, uint32_t>;

// Verification Tag, used for packet validation.
using VerificationTag = webrtc::StrongAlias<class VerificationTagTag, uint32_t>;

// Tie Tag, used as a nonce when connecting.
using TieTag = webrtc::StrongAlias<class TieTagTag, uint64_t>;

// An ID for every outgoing message, to correlate outgoing data chunks with the
// message it was carved from.
using OutgoingMessageId =
    webrtc::StrongAlias<class OutgoingMessageIdTag, uint32_t>;

}  // namespace dcsctp
#endif  // NET_DCSCTP_COMMON_INTERNAL_TYPES_H_
