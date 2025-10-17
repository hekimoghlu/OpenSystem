/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 2, 2025.
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
#ifndef RTC_BASE_NETWORK_ROUTE_H_
#define RTC_BASE_NETWORK_ROUTE_H_

#include <stdint.h>

#include <string>

#include "rtc_base/network_constants.h"
#include "rtc_base/strings/string_builder.h"
#include "rtc_base/system/inline.h"

// TODO(honghaiz): Make a directory that describes the interfaces and structs
// the media code can rely on and the network code can implement, and both can
// depend on that, but not depend on each other. Then, move this file to that
// directory.
namespace rtc {

class RouteEndpoint {
 public:
  RouteEndpoint() {}  // Used by tests.
  RouteEndpoint(AdapterType adapter_type,
                uint16_t adapter_id,
                uint16_t network_id,
                bool uses_turn)
      : adapter_type_(adapter_type),
        adapter_id_(adapter_id),
        network_id_(network_id),
        uses_turn_(uses_turn) {}

  RouteEndpoint(const RouteEndpoint&) = default;
  RouteEndpoint& operator=(const RouteEndpoint&) = default;

  // Used by tests.
  static RouteEndpoint CreateWithNetworkId(uint16_t network_id) {
    return RouteEndpoint(ADAPTER_TYPE_UNKNOWN,
                         /* adapter_id = */ 0, network_id,
                         /* uses_turn = */ false);
  }
  RouteEndpoint CreateWithTurn(bool uses_turn) const {
    return RouteEndpoint(adapter_type_, adapter_id_, network_id_, uses_turn);
  }

  AdapterType adapter_type() const { return adapter_type_; }
  uint16_t adapter_id() const { return adapter_id_; }
  uint16_t network_id() const { return network_id_; }
  bool uses_turn() const { return uses_turn_; }

  bool operator==(const RouteEndpoint& other) const;

 private:
  AdapterType adapter_type_ = ADAPTER_TYPE_UNKNOWN;
  uint16_t adapter_id_ = 0;
  uint16_t network_id_ = 0;
  bool uses_turn_ = false;
};

struct NetworkRoute {
  bool connected = false;
  RouteEndpoint local;
  RouteEndpoint remote;
  // Last packet id sent on the PREVIOUS route.
  int last_sent_packet_id = -1;
  // The overhead in bytes from IP layer and above.
  // This is the maximum of any part of the route.
  int packet_overhead = 0;

  RTC_NO_INLINE inline std::string DebugString() const {
    rtc::StringBuilder oss;
    oss << "[ connected: " << connected << " local: [ " << local.adapter_id()
        << "/" << local.network_id() << " "
        << AdapterTypeToString(local.adapter_type())
        << " turn: " << local.uses_turn() << " ] remote: [ "
        << remote.adapter_id() << "/" << remote.network_id() << " "
        << AdapterTypeToString(remote.adapter_type())
        << " turn: " << remote.uses_turn()
        << " ] packet_overhead_bytes: " << packet_overhead << " ]";
    return oss.Release();
  }

  bool operator==(const NetworkRoute& other) const;
  bool operator!=(const NetworkRoute& other) { return !operator==(other); }
};

}  // namespace rtc

#endif  // RTC_BASE_NETWORK_ROUTE_H_
