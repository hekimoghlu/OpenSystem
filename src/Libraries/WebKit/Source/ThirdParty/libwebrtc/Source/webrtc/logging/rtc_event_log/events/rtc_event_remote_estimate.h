/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 5, 2024.
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
#ifndef LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_REMOTE_ESTIMATE_H_
#define LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_REMOTE_ESTIMATE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/rtc_event_log/rtc_event.h"
#include "api/units/data_rate.h"
#include "api/units/timestamp.h"
#include "logging/rtc_event_log/events/rtc_event_log_parse_status.h"

namespace webrtc {

struct LoggedRemoteEstimateEvent {
  LoggedRemoteEstimateEvent() = default;

  int64_t log_time_us() const { return timestamp.us(); }
  int64_t log_time_ms() const { return timestamp.ms(); }
  Timestamp log_time() const { return timestamp; }

  Timestamp timestamp = Timestamp::MinusInfinity();
  std::optional<DataRate> link_capacity_lower;
  std::optional<DataRate> link_capacity_upper;
};

class RtcEventRemoteEstimate final : public RtcEvent {
 public:
  static constexpr Type kType = Type::RemoteEstimateEvent;

  RtcEventRemoteEstimate(DataRate link_capacity_lower,
                         DataRate link_capacity_upper)
      : link_capacity_lower_(link_capacity_lower),
        link_capacity_upper_(link_capacity_upper) {}

  Type GetType() const override { return kType; }
  bool IsConfigEvent() const override { return false; }

  static std::string Encode(rtc::ArrayView<const RtcEvent*> /* batch */) {
    // TODO(terelius): Implement
    return "";
  }

  static RtcEventLogParseStatus Parse(
      absl::string_view /* encoded_bytes */,
      bool /* batched */,
      std::vector<LoggedRemoteEstimateEvent>& /* output */) {
    // TODO(terelius): Implement
    return RtcEventLogParseStatus::Error("Not Implemented", __FILE__, __LINE__);
  }

  const DataRate link_capacity_lower_;
  const DataRate link_capacity_upper_;
};

}  // namespace webrtc
#endif  // LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_REMOTE_ESTIMATE_H_
