/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 7, 2021.
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
#include "logging/rtc_event_log/events/rtc_event_ice_candidate_pair.h"

#include <cstdint>
#include <memory>

#include "absl/memory/memory.h"
#include "api/rtc_event_log/rtc_event.h"

namespace webrtc {

RtcEventIceCandidatePair::RtcEventIceCandidatePair(
    IceCandidatePairEventType type,
    uint32_t candidate_pair_id,
    uint32_t transaction_id)
    : type_(type),
      candidate_pair_id_(candidate_pair_id),
      transaction_id_(transaction_id) {}

RtcEventIceCandidatePair::RtcEventIceCandidatePair(
    const RtcEventIceCandidatePair& other)
    : RtcEvent(other.timestamp_us_),
      type_(other.type_),
      candidate_pair_id_(other.candidate_pair_id_),
      transaction_id_(other.transaction_id_) {}

RtcEventIceCandidatePair::~RtcEventIceCandidatePair() = default;

std::unique_ptr<RtcEventIceCandidatePair> RtcEventIceCandidatePair::Copy()
    const {
  return absl::WrapUnique<RtcEventIceCandidatePair>(
      new RtcEventIceCandidatePair(*this));
}

}  // namespace webrtc
