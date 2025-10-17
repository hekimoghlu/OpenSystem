/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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
#ifndef LOGGING_RTC_EVENT_LOG_ICE_LOGGER_H_
#define LOGGING_RTC_EVENT_LOG_ICE_LOGGER_H_

#include <cstdint>
#include <unordered_map>

#include "logging/rtc_event_log/events/rtc_event_ice_candidate_pair.h"
#include "logging/rtc_event_log/events/rtc_event_ice_candidate_pair_config.h"

namespace webrtc {

class RtcEventLog;

// IceEventLog wraps RtcEventLog and provides structural logging of ICE-specific
// events. The logged events are serialized with other RtcEvent's if protobuf is
// enabled in the build.
class IceEventLog {
 public:
  IceEventLog();
  ~IceEventLog();

  void set_event_log(RtcEventLog* event_log) { event_log_ = event_log; }

  void LogCandidatePairConfig(
      IceCandidatePairConfigType type,
      uint32_t candidate_pair_id,
      const IceCandidatePairDescription& candidate_pair_desc);

  void LogCandidatePairEvent(IceCandidatePairEventType type,
                             uint32_t candidate_pair_id,
                             uint32_t transaction_id);

  // This method constructs a config event for each candidate pair with their
  // description and logs these config events. It is intended to be called when
  // logging starts to ensure that we have at least one config for each
  // candidate pair id.
  void DumpCandidatePairDescriptionToMemoryAsConfigEvents() const;

 private:
  RtcEventLog* event_log_ = nullptr;
  std::unordered_map<uint32_t, IceCandidatePairDescription>
      candidate_pair_desc_by_id_;
};

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_ICE_LOGGER_H_
