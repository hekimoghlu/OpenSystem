/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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
#ifndef LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_NETEQ_SET_MINIMUM_DELAY_H_
#define LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_NETEQ_SET_MINIMUM_DELAY_H_

#include <stdint.h>

#include <memory>

#include "absl/memory/memory.h"
#include "api/rtc_event_log/rtc_event.h"
#include "api/units/timestamp.h"

namespace webrtc {

struct LoggedNetEqSetMinimumDelayEvent {
  LoggedNetEqSetMinimumDelayEvent() = default;
  LoggedNetEqSetMinimumDelayEvent(Timestamp timestamp,
                                  uint32_t remote_ssrc,
                                  int minimum_delay_ms)
      : timestamp(timestamp),
        remote_ssrc(remote_ssrc),
        minimum_delay_ms(minimum_delay_ms) {}

  int64_t log_time_us() const { return timestamp.us(); }
  int64_t log_time_ms() const { return timestamp.ms(); }
  Timestamp log_time() const { return timestamp; }

  Timestamp timestamp = Timestamp::MinusInfinity();
  uint32_t remote_ssrc;
  int minimum_delay_ms;
};

class RtcEventNetEqSetMinimumDelay final : public RtcEvent {
 public:
  static constexpr Type kType = Type::NetEqSetMinimumDelay;

  explicit RtcEventNetEqSetMinimumDelay(uint32_t remote_ssrc, int delay_ms);
  ~RtcEventNetEqSetMinimumDelay() override;

  Type GetType() const override { return kType; }
  bool IsConfigEvent() const override { return false; }
  uint32_t remote_ssrc() const { return remote_ssrc_; }
  int minimum_delay_ms() const { return minimum_delay_ms_; }

  std::unique_ptr<RtcEventNetEqSetMinimumDelay> Copy() const {
    return absl::WrapUnique<RtcEventNetEqSetMinimumDelay>(
        new RtcEventNetEqSetMinimumDelay(*this));
  }

 private:
  uint32_t remote_ssrc_;
  int minimum_delay_ms_;
};

}  // namespace webrtc
#endif  // LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_NETEQ_SET_MINIMUM_DELAY_H_
