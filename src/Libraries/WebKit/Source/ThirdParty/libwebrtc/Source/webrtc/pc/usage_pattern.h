/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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
#ifndef PC_USAGE_PATTERN_H_
#define PC_USAGE_PATTERN_H_

#include "api/peer_connection_interface.h"

namespace webrtc {

class PeerConnectionObserver;

// A bit in the usage pattern is registered when its defining event occurs
// at least once.
enum class UsageEvent : int {
  TURN_SERVER_ADDED = 0x01,
  STUN_SERVER_ADDED = 0x02,
  DATA_ADDED = 0x04,
  AUDIO_ADDED = 0x08,
  VIDEO_ADDED = 0x10,
  // `SetLocalDescription` returns successfully.
  SET_LOCAL_DESCRIPTION_SUCCEEDED = 0x20,
  // `SetRemoteDescription` returns successfully.
  SET_REMOTE_DESCRIPTION_SUCCEEDED = 0x40,
  // A local candidate (with type host, server-reflexive, or relay) is
  // collected.
  CANDIDATE_COLLECTED = 0x80,
  // A remote candidate is successfully added via `AddIceCandidate`.
  ADD_ICE_CANDIDATE_SUCCEEDED = 0x100,
  ICE_STATE_CONNECTED = 0x200,
  CLOSE_CALLED = 0x400,
  // A local candidate with private IP is collected.
  PRIVATE_CANDIDATE_COLLECTED = 0x800,
  // A remote candidate with private IP is added, either via AddiceCandidate
  // or from the remote description.
  REMOTE_PRIVATE_CANDIDATE_ADDED = 0x1000,
  // A local mDNS candidate is collected.
  MDNS_CANDIDATE_COLLECTED = 0x2000,
  // A remote mDNS candidate is added, either via AddIceCandidate or from the
  // remote description.
  REMOTE_MDNS_CANDIDATE_ADDED = 0x4000,
  // A local candidate with IPv6 address is collected.
  IPV6_CANDIDATE_COLLECTED = 0x8000,
  // A remote candidate with IPv6 address is added, either via AddIceCandidate
  // or from the remote description.
  REMOTE_IPV6_CANDIDATE_ADDED = 0x10000,
  // A remote candidate (with type host, server-reflexive, or relay) is
  // successfully added, either via AddIceCandidate or from the remote
  // description.
  REMOTE_CANDIDATE_ADDED = 0x20000,
  // An explicit host-host candidate pair is selected, i.e. both the local and
  // the remote candidates have the host type. This does not include candidate
  // pairs formed with equivalent prflx remote candidates, e.g. a host-prflx
  // pair where the prflx candidate has the same base as a host candidate of
  // the remote peer.
  DIRECT_CONNECTION_SELECTED = 0x40000,
  MAX_VALUE = 0x80000,
};

class UsagePattern {
 public:
  void NoteUsageEvent(UsageEvent event);
  void ReportUsagePattern(PeerConnectionObserver* observer) const;

 private:
  int usage_event_accumulator_ = 0;
};

}  // namespace webrtc
#endif  // PC_USAGE_PATTERN_H_
