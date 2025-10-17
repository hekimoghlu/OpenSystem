/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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
#include "logging/rtc_event_log/ice_logger.h"

#include <cstdint>
#include <memory>

#include "api/rtc_event_log/rtc_event_log.h"
#include "logging/rtc_event_log/events/rtc_event_ice_candidate_pair.h"
#include "logging/rtc_event_log/events/rtc_event_ice_candidate_pair_config.h"

namespace webrtc {

IceEventLog::IceEventLog() {}
IceEventLog::~IceEventLog() {}

void IceEventLog::LogCandidatePairConfig(
    IceCandidatePairConfigType type,
    uint32_t candidate_pair_id,
    const IceCandidatePairDescription& candidate_pair_desc) {
  if (event_log_ == nullptr) {
    return;
  }

  candidate_pair_desc_by_id_.emplace(candidate_pair_id, candidate_pair_desc);
  event_log_->Log(std::make_unique<RtcEventIceCandidatePairConfig>(
      type, candidate_pair_id, candidate_pair_desc));
}

void IceEventLog::LogCandidatePairEvent(IceCandidatePairEventType type,
                                        uint32_t candidate_pair_id,
                                        uint32_t transaction_id) {
  if (event_log_ == nullptr) {
    return;
  }
  event_log_->Log(std::make_unique<RtcEventIceCandidatePair>(
      type, candidate_pair_id, transaction_id));
}

void IceEventLog::DumpCandidatePairDescriptionToMemoryAsConfigEvents() const {
  for (const auto& desc_id_pair : candidate_pair_desc_by_id_) {
    event_log_->Log(std::make_unique<RtcEventIceCandidatePairConfig>(
        IceCandidatePairConfigType::kUpdated, desc_id_pair.first,
        desc_id_pair.second));
  }
}

}  // namespace webrtc
