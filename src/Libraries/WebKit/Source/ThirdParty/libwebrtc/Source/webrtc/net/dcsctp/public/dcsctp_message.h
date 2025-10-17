/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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
#ifndef NET_DCSCTP_PUBLIC_DCSCTP_MESSAGE_H_
#define NET_DCSCTP_PUBLIC_DCSCTP_MESSAGE_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "api/array_view.h"
#include "net/dcsctp/public/types.h"

namespace dcsctp {

// An SCTP message is a group of bytes sent and received as a whole on a
// specified stream identifier (`stream_id`), and with a payload protocol
// identifier (`ppid`).
class DcSctpMessage {
 public:
  DcSctpMessage(StreamID stream_id, PPID ppid, std::vector<uint8_t> payload)
      : stream_id_(stream_id), ppid_(ppid), payload_(std::move(payload)) {}

  DcSctpMessage(DcSctpMessage&& other) = default;
  DcSctpMessage& operator=(DcSctpMessage&& other) = default;
  DcSctpMessage(const DcSctpMessage&) = delete;
  DcSctpMessage& operator=(const DcSctpMessage&) = delete;

  // The stream identifier to which the message is sent.
  StreamID stream_id() const { return stream_id_; }

  // The payload protocol identifier (ppid) associated with the message.
  PPID ppid() const { return ppid_; }

  // The payload of the message.
  rtc::ArrayView<const uint8_t> payload() const { return payload_; }

  // When destructing the message, extracts the payload.
  std::vector<uint8_t> ReleasePayload() && { return std::move(payload_); }

 private:
  StreamID stream_id_;
  PPID ppid_;
  std::vector<uint8_t> payload_;
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_PUBLIC_DCSCTP_MESSAGE_H_
