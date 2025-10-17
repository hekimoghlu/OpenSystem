/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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
#ifndef LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_AUDIO_RECEIVE_STREAM_CONFIG_H_
#define LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_AUDIO_RECEIVE_STREAM_CONFIG_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/rtc_event_log/rtc_event.h"
#include "api/units/timestamp.h"
#include "logging/rtc_event_log/events/rtc_event_log_parse_status.h"
#include "logging/rtc_event_log/rtc_stream_config.h"

namespace webrtc {

struct LoggedAudioRecvConfig {
  LoggedAudioRecvConfig() = default;
  LoggedAudioRecvConfig(Timestamp timestamp, const rtclog::StreamConfig config)
      : timestamp(timestamp), config(config) {}

  int64_t log_time_us() const { return timestamp.us(); }
  int64_t log_time_ms() const { return timestamp.ms(); }
  Timestamp log_time() const { return timestamp; }

  Timestamp timestamp = Timestamp::MinusInfinity();
  rtclog::StreamConfig config;
};

class RtcEventAudioReceiveStreamConfig final : public RtcEvent {
 public:
  static constexpr Type kType = Type::AudioReceiveStreamConfig;

  explicit RtcEventAudioReceiveStreamConfig(
      std::unique_ptr<rtclog::StreamConfig> config);
  ~RtcEventAudioReceiveStreamConfig() override;

  Type GetType() const override { return kType; }
  bool IsConfigEvent() const override { return true; }

  std::unique_ptr<RtcEventAudioReceiveStreamConfig> Copy() const;

  const rtclog::StreamConfig& config() const { return *config_; }

  static std::string Encode(rtc::ArrayView<const RtcEvent*> /* batch */) {
    // TODO(terelius): Implement
    return "";
  }

  static RtcEventLogParseStatus Parse(
      absl::string_view /* encoded_bytes */,
      bool /* batched */,
      std::vector<LoggedAudioRecvConfig>& /* output */) {
    // TODO(terelius): Implement
    return RtcEventLogParseStatus::Error("Not Implemented", __FILE__, __LINE__);
  }

 private:
  RtcEventAudioReceiveStreamConfig(
      const RtcEventAudioReceiveStreamConfig& other);

  const std::unique_ptr<const rtclog::StreamConfig> config_;
};

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_AUDIO_RECEIVE_STREAM_CONFIG_H_
