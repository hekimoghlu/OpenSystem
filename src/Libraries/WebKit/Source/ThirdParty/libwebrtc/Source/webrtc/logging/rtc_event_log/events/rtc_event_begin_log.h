/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#ifndef LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_BEGIN_LOG_H_
#define LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_BEGIN_LOG_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/rtc_event_log/rtc_event.h"
#include "api/units/timestamp.h"
#include "logging/rtc_event_log/events/rtc_event_field_encoding.h"
#include "logging/rtc_event_log/events/rtc_event_log_parse_status.h"

namespace webrtc {

struct LoggedStartEvent {
  LoggedStartEvent() = default;

  explicit LoggedStartEvent(Timestamp timestamp)
      : LoggedStartEvent(timestamp, timestamp) {}

  LoggedStartEvent(Timestamp timestamp, Timestamp utc_start_time)
      : timestamp(timestamp), utc_start_time(utc_start_time) {}

  int64_t log_time_us() const { return timestamp.us(); }
  int64_t log_time_ms() const { return timestamp.ms(); }
  Timestamp log_time() const { return timestamp; }

  Timestamp utc_time() const { return utc_start_time; }

  Timestamp timestamp = Timestamp::PlusInfinity();
  Timestamp utc_start_time = Timestamp::PlusInfinity();
};

class RtcEventBeginLog final : public RtcEvent {
 public:
  static constexpr Type kType = Type::BeginV3Log;

  RtcEventBeginLog(Timestamp timestamp, Timestamp utc_start_time);
  ~RtcEventBeginLog() override;

  Type GetType() const override { return kType; }
  bool IsConfigEvent() const override { return false; }

  static std::string Encode(rtc::ArrayView<const RtcEvent*> batch);

  static RtcEventLogParseStatus Parse(absl::string_view encoded_bytes,
                                      bool batched,
                                      std::vector<LoggedStartEvent>& output);

 private:
  RtcEventBeginLog(const RtcEventBeginLog& other);

  int64_t utc_start_time_ms_;

  static constexpr EventParameters event_params_{"BeginLog",
                                                 RtcEventBeginLog::kType};
  static constexpr FieldParameters utc_start_time_params_{
      "utc_start_time_ms", /*id=*/1, FieldType::kVarInt, /*width=*/64};
};

}  // namespace webrtc
#endif  // LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_BEGIN_LOG_H_
