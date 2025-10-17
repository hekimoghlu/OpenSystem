/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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
#ifndef API_TRANSPORT_ENUMS_H_
#define API_TRANSPORT_ENUMS_H_

namespace webrtc {

// See https://w3c.github.io/webrtc-pc/#rtcicetransportstate
// Note that kFailed is currently not a terminal state, and a transport might
// incorrectly be marked as failed while gathering candidates, see
// bugs.webrtc.org/8833
enum class IceTransportState {
  kNew,
  kChecking,
  kConnected,
  kCompleted,
  kFailed,
  kDisconnected,
  kClosed,
};

enum PortPrunePolicy {
  NO_PRUNE,                 // Do not prune.
  PRUNE_BASED_ON_PRIORITY,  // Prune lower-priority ports on the same network.
  KEEP_FIRST_READY          // Keep the first ready port and prune the rest
                            // on the same network.
};

enum class VpnPreference {
  kDefault,      // No VPN preference.
  kOnlyUseVpn,   // only use VPN connections.
  kNeverUseVpn,  // never use VPN connections
  kPreferVpn,    // use a VPN connection if possible,
                 // i.e VPN connections sorts first.
  kAvoidVpn,     // only use VPN if there is no other connections,
                 // i.e VPN connections sorts last.
};

}  // namespace webrtc

#endif  // API_TRANSPORT_ENUMS_H_
