/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 5, 2022.
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
#ifndef LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_AUDIO_PLAYOUT_H_
#define LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_AUDIO_PLAYOUT_H_

#include <stdint.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/rtc_event_log/rtc_event.h"
#include "api/units/timestamp.h"
#include "logging/rtc_event_log/events/rtc_event_definition.h"
#include "logging/rtc_event_log/events/rtc_event_field_encoding.h"
#include "logging/rtc_event_log/events/rtc_event_log_parse_status.h"

namespace webrtc {

struct LoggedAudioPlayoutEvent {
  LoggedAudioPlayoutEvent() = default;
  LoggedAudioPlayoutEvent(Timestamp timestamp, uint32_t ssrc)
      : timestamp(timestamp), ssrc(ssrc) {}

  int64_t log_time_us() const { return timestamp.us(); }
  int64_t log_time_ms() const { return timestamp.ms(); }
  Timestamp log_time() const { return timestamp; }

  Timestamp timestamp = Timestamp::MinusInfinity();
  uint32_t ssrc;
};

class RtcEventAudioPlayout final : public RtcEvent {
 public:
  static constexpr Type kType = Type::AudioPlayout;

  explicit RtcEventAudioPlayout(uint32_t ssrc);
  ~RtcEventAudioPlayout() override = default;

  Type GetType() const override { return kType; }
  bool IsConfigEvent() const override { return false; }

  std::unique_ptr<RtcEventAudioPlayout> Copy() const;

  uint32_t ssrc() const { return ssrc_; }

  static std::string Encode(rtc::ArrayView<const RtcEvent*> batch) {
    return RtcEventAudioPlayout::definition_.EncodeBatch(batch);
  }

  static RtcEventLogParseStatus Parse(
      absl::string_view encoded_bytes,
      bool batched,
      std::map<uint32_t, std::vector<LoggedAudioPlayoutEvent>>& output) {
    std::vector<LoggedAudioPlayoutEvent> temp_output;
    auto status = RtcEventAudioPlayout::definition_.ParseBatch(
        encoded_bytes, batched, temp_output);
    for (const LoggedAudioPlayoutEvent& event : temp_output) {
      output[event.ssrc].push_back(event);
    }
    return status;
  }

 private:
  RtcEventAudioPlayout(const RtcEventAudioPlayout& other);

  const uint32_t ssrc_;

  static constexpr RtcEventDefinition<RtcEventAudioPlayout,
                                      LoggedAudioPlayoutEvent,
                                      uint32_t>
      definition_{{"AudioPlayout", RtcEventAudioPlayout::kType},
                  {&RtcEventAudioPlayout::ssrc_,
                   &LoggedAudioPlayoutEvent::ssrc,
                   {"ssrc", /*id=*/1, FieldType::kFixed32, /*width=*/32}}};
};

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_AUDIO_PLAYOUT_H_
