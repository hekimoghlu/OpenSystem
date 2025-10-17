/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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
#ifndef LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_PROBE_RESULT_FAILURE_H_
#define LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_PROBE_RESULT_FAILURE_H_

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/rtc_event_log/rtc_event.h"
#include "api/units/timestamp.h"
#include "logging/rtc_event_log/events/rtc_event_log_parse_status.h"

namespace webrtc {

enum class ProbeFailureReason {
  kInvalidSendReceiveInterval = 0,
  kInvalidSendReceiveRatio,
  kTimeout,
  kLast
};

struct LoggedBweProbeFailureEvent {
  LoggedBweProbeFailureEvent() = default;
  LoggedBweProbeFailureEvent(Timestamp timestamp,
                             int32_t id,
                             ProbeFailureReason failure_reason)
      : timestamp(timestamp), id(id), failure_reason(failure_reason) {}

  int64_t log_time_us() const { return timestamp.us(); }
  int64_t log_time_ms() const { return timestamp.ms(); }
  Timestamp log_time() const { return timestamp; }

  Timestamp timestamp = Timestamp::MinusInfinity();
  int32_t id;
  ProbeFailureReason failure_reason;
};

class RtcEventProbeResultFailure final : public RtcEvent {
 public:
  static constexpr Type kType = Type::ProbeResultFailure;

  RtcEventProbeResultFailure(int32_t id, ProbeFailureReason failure_reason);
  ~RtcEventProbeResultFailure() override = default;

  Type GetType() const override { return kType; }
  bool IsConfigEvent() const override { return false; }

  std::unique_ptr<RtcEventProbeResultFailure> Copy() const;

  int32_t id() const { return id_; }
  ProbeFailureReason failure_reason() const { return failure_reason_; }

  static std::string Encode(rtc::ArrayView<const RtcEvent*> /* batch */) {
    // TODO(terelius): Implement
    return "";
  }

  static RtcEventLogParseStatus Parse(
      absl::string_view /* encoded_bytes */,
      bool /* batched */,
      std::vector<LoggedBweProbeFailureEvent>& /* output */) {
    // TODO(terelius): Implement
    return RtcEventLogParseStatus::Error("Not Implemented", __FILE__, __LINE__);
  }

 private:
  RtcEventProbeResultFailure(const RtcEventProbeResultFailure& other);

  const int32_t id_;
  const ProbeFailureReason failure_reason_;
};

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_PROBE_RESULT_FAILURE_H_
