/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 19, 2023.
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
#include "logging/rtc_event_log/events/rtc_event_end_log.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/rtc_event_log/rtc_event.h"
#include "api/units/timestamp.h"
#include "logging/rtc_event_log/events/rtc_event_field_encoding.h"
#include "logging/rtc_event_log/events/rtc_event_field_encoding_parser.h"
#include "logging/rtc_event_log/events/rtc_event_log_parse_status.h"

namespace webrtc {
constexpr RtcEvent::Type RtcEventEndLog::kType;
constexpr EventParameters RtcEventEndLog::event_params_;

RtcEventEndLog::RtcEventEndLog(Timestamp timestamp)
    : RtcEvent(timestamp.us()) {}

RtcEventEndLog::RtcEventEndLog(const RtcEventEndLog& other)
    : RtcEvent(other.timestamp_us_) {}

RtcEventEndLog::~RtcEventEndLog() = default;

std::string RtcEventEndLog::Encode(rtc::ArrayView<const RtcEvent*> batch) {
  EventEncoder encoder(event_params_, batch);
  return encoder.AsString();
}

RtcEventLogParseStatus RtcEventEndLog::Parse(
    absl::string_view encoded_bytes,
    bool batched,
    std::vector<LoggedStopEvent>& output) {
  EventParser parser;
  auto status = parser.Initialize(encoded_bytes, batched);
  if (!status.ok())
    return status;

  rtc::ArrayView<LoggedStopEvent> output_batch =
      ExtendLoggedBatch(output, parser.NumEventsInBatch());

  constexpr FieldParameters timestamp_params{
      "timestamp_ms", FieldParameters::kTimestampField, FieldType::kVarInt, 64};
  RtcEventLogParseStatusOr<rtc::ArrayView<uint64_t>> result =
      parser.ParseNumericField(timestamp_params);
  if (!result.ok())
    return result.status();
  status = PopulateRtcEventTimestamp(result.value(),
                                     &LoggedStopEvent::timestamp, output_batch);
  if (!status.ok())
    return status;

  return RtcEventLogParseStatus::Success();
}

}  // namespace webrtc
