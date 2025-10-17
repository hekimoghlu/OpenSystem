/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
#ifndef NET_DCSCTP_SOCKET_HEARTBEAT_HANDLER_H_
#define NET_DCSCTP_SOCKET_HEARTBEAT_HANDLER_H_

#include <stdint.h>

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "net/dcsctp/packet/chunk/heartbeat_ack_chunk.h"
#include "net/dcsctp/packet/chunk/heartbeat_request_chunk.h"
#include "net/dcsctp/packet/sctp_packet.h"
#include "net/dcsctp/public/dcsctp_options.h"
#include "net/dcsctp/socket/context.h"
#include "net/dcsctp/timer/timer.h"

namespace dcsctp {

// HeartbeatHandler handles all logic around sending heartbeats and receiving
// the responses, as well as receiving incoming heartbeat requests.
//
// Heartbeats are sent on idle connections to ensure that the connection is
// still healthy and to measure the RTT. If a number of heartbeats time out,
// the connection will eventually be closed.
class HeartbeatHandler {
 public:
  HeartbeatHandler(absl::string_view log_prefix,
                   const DcSctpOptions& options,
                   Context* context,
                   TimerManager* timer_manager);

  // Called when the heartbeat interval timer should be restarted. This is
  // generally done every time data is sent, which makes the timer expire when
  // the connection is idle.
  void RestartTimer();

  // Called on received HeartbeatRequestChunk chunks.
  void HandleHeartbeatRequest(HeartbeatRequestChunk chunk);

  // Called on received HeartbeatRequestChunk chunks.
  void HandleHeartbeatAck(HeartbeatAckChunk chunk);

 private:
  webrtc::TimeDelta OnIntervalTimerExpiry();
  webrtc::TimeDelta OnTimeoutTimerExpiry();

  const absl::string_view log_prefix_;
  Context* ctx_;
  TimerManager* timer_manager_;
  // The time for a connection to be idle before a heartbeat is sent.
  const webrtc::TimeDelta interval_duration_;
  // Adding RTT to the duration will add some jitter, which is good in
  // production, but less good in unit tests, which is why it can be disabled.
  const bool interval_duration_should_include_rtt_;
  const std::unique_ptr<Timer> interval_timer_;
  const std::unique_ptr<Timer> timeout_timer_;
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_SOCKET_HEARTBEAT_HANDLER_H_
