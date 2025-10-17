/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 27, 2024.
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
#ifndef NET_DCSCTP_SOCKET_CAPABILITIES_H_
#define NET_DCSCTP_SOCKET_CAPABILITIES_H_

#include <cstdint>
namespace dcsctp {
// Indicates what the association supports, meaning that both parties
// support it and that feature can be used.
struct Capabilities {
  // RFC3758 Partial Reliability Extension
  bool partial_reliability = false;
  // RFC8260 Stream Schedulers and User Message Interleaving
  bool message_interleaving = false;
  // RFC6525 Stream Reconfiguration
  bool reconfig = false;
  // https://datatracker.ietf.org/doc/draft-ietf-tsvwg-sctp-zero-checksum/
  bool zero_checksum = false;
  // Negotiated maximum incoming and outgoing stream count.
  uint16_t negotiated_maximum_incoming_streams = 0;
  uint16_t negotiated_maximum_outgoing_streams = 0;
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_SOCKET_CAPABILITIES_H_
