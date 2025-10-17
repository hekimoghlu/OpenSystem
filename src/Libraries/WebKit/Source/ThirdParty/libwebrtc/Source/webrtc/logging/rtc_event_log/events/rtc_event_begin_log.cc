/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 3, 2024.
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
#include "logging/rtc_event_log/events/rtc_event_begin_log.h"

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
constexpr RtcEvent::Type RtcEventBeginLog::kType;
constexpr EventParameters RtcEventBeginLog::event_params_;
constexpr FieldParameters RtcEventBeginLog::utc_start_time_params_;

RtcEventBeginLog::RtcEventBeginLog(Timestamp timestamp,
                                   Timestamp utc_start_time)
    : RtcEvent(timestamp.us()), utc_start_time_ms_(utc_start_time.ms()) {}

RtcEventBeginLog::RtcEventBeginLog(const RtcEventBeginLog& other)
    : RtcEvent(other.timestamp_us_) {}

RtcEventBeginLog::~RtcEventBeginLog() = default;

std::string RtcEventBeginLog::Encode(rtc::ArrayView<const RtcEvent*> batch) {
  EventEncoder encoder(event_params_, batch);

  encoder.EncodeField(
      utc_start_time_params_,
      ExtractRtcEventMember(batch, &RtcEventBeginLog::utc_start_time_ms_));

  return encoder.AsString();
}

RtcEventLogParseStatus RtcEventBeginLog::Parse(
    absl::string_view encoded_bytes,
    bool batched,
    std::vector<LoggedStartEvent>& output) {
  EventParser parser;
  auto status = parser.Initialize(encoded_bytes, batched);
  if (!status.ok())
    return status;

  rtc::ArrayView<LoggedStartEvent> output_batch =
      ExtendLoggedBatch(output, parser.NumEventsInBatch());

  constexpr FieldParameters timestamp_params{
      "timestamp_ms", FieldParameters::kTimestampField, FieldType::kVarInt, 64};
  RtcEventLogParseStatusOr<rtc::ArrayView<uint64_t>> result =
      parser.ParseNumericField(timestamp_params);
  if (!result.ok())
    return result.status();
  status = PopulateRtcEventTimestamp(
      result.value(), &LoggedStartEvent::timestamp, output_batch);
  if (!status.ok())
    return status;

  result = parser.ParseNumericField(utc_start_time_params_);
  if (!result.ok())
    return result.status();
  status = PopulateRtcEventTimestamp(
      result.value(), &LoggedStartEvent::utc_start_time, output_batch);
  if (!status.ok())
    return status;

  return RtcEventLogParseStatus::Success();
}

}  // namespace webrtc
