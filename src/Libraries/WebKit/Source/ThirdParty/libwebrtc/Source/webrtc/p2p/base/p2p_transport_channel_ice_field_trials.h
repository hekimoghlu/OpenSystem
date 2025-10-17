/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 11, 2025.
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
#ifndef P2P_BASE_P2P_TRANSPORT_CHANNEL_ICE_FIELD_TRIALS_H_
#define P2P_BASE_P2P_TRANSPORT_CHANNEL_ICE_FIELD_TRIALS_H_

#include <optional>

namespace cricket {

// Field trials for P2PTransportChannel and friends,
// put in separate file so that they can be shared e.g
// with Connection.
struct IceFieldTrials {
  // This struct is built using the FieldTrialParser, and then not modified.
  // TODO(jonaso) : Consider how members of this struct can be made const.

  bool skip_relay_to_non_relay_connections = false;
  std::optional<int> max_outstanding_pings;

  // Wait X ms before selecting a connection when having none.
  // This will make media slower, but will give us chance to find
  // a better connection before starting.
  std::optional<int> initial_select_dampening;

  // If the connection has recevied a ping-request, delay by
  // maximum this delay. This will make media slower, but will
  // give us chance to find a better connection before starting.
  std::optional<int> initial_select_dampening_ping_received;

  // Announce GOOG_PING support in STUN_BINDING_RESPONSE if requested
  // by peer.
  bool announce_goog_ping = true;

  // Enable sending GOOG_PING if remote announce it.
  bool enable_goog_ping = false;

  // Decay rate for RTT estimate using EventBasedExponentialMovingAverage
  // expressed as halving time.
  int rtt_estimate_halftime_ms = 500;

  // Sending a PING directly after a switch on ICE_CONTROLLING-side.
  // TODO(jonaso) : Deprecate this in favor of
  // `send_ping_on_selected_ice_controlling`.
  bool send_ping_on_switch_ice_controlling = false;

  // Sending a PING directly after selecting a connection
  // (i.e either a switch or the inital selection).
  bool send_ping_on_selected_ice_controlling = false;

  // Sending a PING directly after a nomination on ICE_CONTROLLED-side.
  bool send_ping_on_nomination_ice_controlled = false;

  // The timeout after which the connection will be considered dead if no
  // traffic is received.
  int dead_connection_timeout_ms = 30000;

  // Stop gathering when having a strong connection.
  bool stop_gather_on_strongly_connected = true;

  // DSCP taging.
  std::optional<int> override_dscp;

  bool piggyback_ice_check_acknowledgement = false;
  bool extra_ice_ping = false;

  // Announce/enable GOOG_DELTA
  bool enable_goog_delta = true;  // send GOOG DELTA
  bool answer_goog_delta = true;  // answer GOOG DELTA
};

}  // namespace cricket

#endif  // P2P_BASE_P2P_TRANSPORT_CHANNEL_ICE_FIELD_TRIALS_H_
