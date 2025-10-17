/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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
#ifndef P2P_BASE_CONNECTION_INFO_H_
#define P2P_BASE_CONNECTION_INFO_H_

#include <optional>
#include <vector>

#include "api/candidate.h"
#include "api/units/timestamp.h"

namespace cricket {

// States are from RFC 5245. http://tools.ietf.org/html/rfc5245#section-5.7.4
enum class IceCandidatePairState {
  WAITING = 0,  // Check has not been performed, Waiting pair on CL.
  IN_PROGRESS,  // Check has been sent, transaction is in progress.
  SUCCEEDED,    // Check already done, produced a successful result.
  FAILED,       // Check for this connection failed.
  // According to spec there should also be a frozen state, but nothing is ever
  // frozen because we have not implemented ICE freezing logic.
};

// Stats that we can return about the connections for a transport channel.
// TODO(hta): Rename to ConnectionStats
struct ConnectionInfo {
  ConnectionInfo();
  ConnectionInfo(const ConnectionInfo&);
  ~ConnectionInfo();

  bool best_connection;         // Is this the best connection we have?
  bool writable;                // Has this connection received a STUN response?
  bool receiving;               // Has this connection received anything?
  bool timeout;                 // Has this connection timed out?
  size_t rtt;                   // The STUN RTT for this connection.
  size_t sent_discarded_bytes;  // Number of outgoing bytes discarded due to
                                // socket errors.
  size_t sent_total_bytes;      // Total bytes sent on this connection. Does not
                                // include discarded bytes.
  size_t sent_bytes_second;     // Bps over the last measurement interval.
  size_t sent_discarded_packets;  // Number of outgoing packets discarded due to
                                  // socket errors.
  size_t sent_total_packets;  // Number of total outgoing packets attempted for
                              // sending, including discarded packets.
  size_t sent_ping_requests_total;  // Number of STUN ping request sent.
  size_t sent_ping_requests_before_first_response;  // Number of STUN ping
  // sent before receiving the first response.
  size_t sent_ping_responses;  // Number of STUN ping response sent.

  size_t recv_total_bytes;     // Total bytes received on this connection.
  size_t recv_bytes_second;    // Bps over the last measurement interval.
  size_t packets_received;     // Number of packets that were received.
  size_t recv_ping_requests;   // Number of STUN ping request received.
  size_t recv_ping_responses;  // Number of STUN ping response received.
  Candidate local_candidate;   // The local candidate for this connection.
  Candidate remote_candidate;  // The remote candidate for this connection.
  void* key;                   // A static value that identifies this conn.
  // https://w3c.github.io/webrtc-stats/#dom-rtcicecandidatepairstats-state
  IceCandidatePairState state;
  // https://w3c.github.io/webrtc-stats/#dom-rtcicecandidatepairstats-priority
  uint64_t priority;
  // https://w3c.github.io/webrtc-stats/#dom-rtcicecandidatepairstats-nominated
  bool nominated;
  // https://w3c.github.io/webrtc-stats/#dom-rtcicecandidatepairstats-totalroundtriptime
  uint64_t total_round_trip_time_ms;
  // https://w3c.github.io/webrtc-stats/#dom-rtcicecandidatepairstats-currentroundtriptime
  std::optional<uint32_t> current_round_trip_time_ms;

  // https://w3c.github.io/webrtc-stats/#dom-rtcicecandidatepairstats-lastpacketreceivedtimestamp
  std::optional<webrtc::Timestamp> last_data_received;
  std::optional<webrtc::Timestamp> last_data_sent;
};

// Information about all the candidate pairs of a channel.
typedef std::vector<ConnectionInfo> ConnectionInfos;

}  // namespace cricket

#endif  // P2P_BASE_CONNECTION_INFO_H_
