/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#ifndef CALL_SIMULATED_PACKET_RECEIVER_H_
#define CALL_SIMULATED_PACKET_RECEIVER_H_

#include <cstdint>
#include <optional>

#include "call/packet_receiver.h"

namespace webrtc {

// Private API that is fixing surface between DirectTransport and underlying
// network conditions simulation implementation.
class SimulatedPacketReceiverInterface : public PacketReceiver {
 public:
  // Must not be called in parallel with DeliverPacket or Process.
  // Destination receiver will be injected with this method
  virtual void SetReceiver(PacketReceiver* receiver) = 0;

  // Reports average packet delay.
  virtual int AverageDelay() = 0;

  // Process any pending tasks such as timeouts.
  // Called on a worker thread.
  virtual void Process() = 0;

  // Returns the time until next process or nullopt to indicate that the next
  // process time is unknown. If the next process time is unknown, this should
  // be checked again any time a packet is enqueued.
  virtual std::optional<int64_t> TimeUntilNextProcess() = 0;
};

}  // namespace webrtc

#endif  // CALL_SIMULATED_PACKET_RECEIVER_H_
