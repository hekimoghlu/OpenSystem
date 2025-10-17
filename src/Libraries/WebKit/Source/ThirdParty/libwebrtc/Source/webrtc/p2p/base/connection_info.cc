/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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
#include "p2p/base/connection_info.h"

namespace cricket {

ConnectionInfo::ConnectionInfo()
    : best_connection(false),
      writable(false),
      receiving(false),
      timeout(false),
      rtt(0),
      sent_discarded_bytes(0),
      sent_total_bytes(0),
      sent_bytes_second(0),
      sent_discarded_packets(0),
      sent_total_packets(0),
      sent_ping_requests_total(0),
      sent_ping_requests_before_first_response(0),
      sent_ping_responses(0),
      recv_total_bytes(0),
      recv_bytes_second(0),
      packets_received(0),
      recv_ping_requests(0),
      recv_ping_responses(0),
      key(nullptr),
      state(IceCandidatePairState::WAITING),
      priority(0),
      nominated(false),
      total_round_trip_time_ms(0) {}

ConnectionInfo::ConnectionInfo(const ConnectionInfo&) = default;

ConnectionInfo::~ConnectionInfo() = default;

}  // namespace cricket
