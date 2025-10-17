/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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
#include "logging/rtc_event_log/events/rtc_event_ice_candidate_pair_config.h"

#include <cstdint>
#include <memory>

#include "absl/memory/memory.h"
#include "api/candidate.h"
#include "api/rtc_event_log/rtc_event.h"

namespace webrtc {

IceCandidatePairDescription::IceCandidatePairDescription(
    IceCandidateType local_candidate_type,
    IceCandidateType remote_candidate_type)
    : local_candidate_type(local_candidate_type),
      remote_candidate_type(remote_candidate_type) {
  local_relay_protocol = IceCandidatePairProtocol::kUnknown;
  local_network_type = IceCandidateNetworkType::kUnknown;
  local_address_family = IceCandidatePairAddressFamily::kUnknown;
  remote_address_family = IceCandidatePairAddressFamily::kUnknown;
  candidate_pair_protocol = IceCandidatePairProtocol::kUnknown;
}

IceCandidatePairDescription::IceCandidatePairDescription(
    const IceCandidatePairDescription& other) {
  local_candidate_type = other.local_candidate_type;
  local_relay_protocol = other.local_relay_protocol;
  local_network_type = other.local_network_type;
  local_address_family = other.local_address_family;
  remote_candidate_type = other.remote_candidate_type;
  remote_address_family = other.remote_address_family;
  candidate_pair_protocol = other.candidate_pair_protocol;
}

IceCandidatePairDescription::~IceCandidatePairDescription() {}

RtcEventIceCandidatePairConfig::RtcEventIceCandidatePairConfig(
    IceCandidatePairConfigType type,
    uint32_t candidate_pair_id,
    const IceCandidatePairDescription& candidate_pair_desc)
    : type_(type),
      candidate_pair_id_(candidate_pair_id),
      candidate_pair_desc_(candidate_pair_desc) {}

RtcEventIceCandidatePairConfig::RtcEventIceCandidatePairConfig(
    const RtcEventIceCandidatePairConfig& other)
    : RtcEvent(other.timestamp_us_),
      type_(other.type_),
      candidate_pair_id_(other.candidate_pair_id_),
      candidate_pair_desc_(other.candidate_pair_desc_) {}

RtcEventIceCandidatePairConfig::~RtcEventIceCandidatePairConfig() = default;

std::unique_ptr<RtcEventIceCandidatePairConfig>
RtcEventIceCandidatePairConfig::Copy() const {
  return absl::WrapUnique<RtcEventIceCandidatePairConfig>(
      new RtcEventIceCandidatePairConfig(*this));
}

}  // namespace webrtc
