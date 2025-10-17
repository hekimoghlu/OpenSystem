/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 23, 2024.
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
#ifndef NET_DCSCTP_PUBLIC_PACKET_OBSERVER_H_
#define NET_DCSCTP_PUBLIC_PACKET_OBSERVER_H_

#include <stdint.h>

#include "api/array_view.h"
#include "net/dcsctp/public/types.h"

namespace dcsctp {

// A PacketObserver can be attached to a socket and will be called for
// all sent and received packets.
class PacketObserver {
 public:
  virtual ~PacketObserver() = default;
  // Called when a packet is sent, with the current time (in milliseconds) as
  // `now`, and the packet payload as `payload`.
  virtual void OnSentPacket(TimeMs now,
                            rtc::ArrayView<const uint8_t> payload) = 0;

  // Called when a packet is received, with the current time (in milliseconds)
  // as `now`, and the packet payload as `payload`.
  virtual void OnReceivedPacket(TimeMs now,
                                rtc::ArrayView<const uint8_t> payload) = 0;
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_PUBLIC_PACKET_OBSERVER_H_
