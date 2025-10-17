/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
#include "api/video/video_timing.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include "api/array_view.h"
#include "api/units/time_delta.h"
#include "rtc_base/logging.h"
#include "rtc_base/numerics/safe_conversions.h"
#include "rtc_base/strings/string_builder.h"

namespace webrtc {

uint16_t VideoSendTiming::GetDeltaCappedMs(int64_t base_ms, int64_t time_ms) {
  if (time_ms < base_ms) {
    RTC_DLOG(LS_ERROR) << "Delta " << (time_ms - base_ms)
                       << "ms expected to be positive";
  }
  return rtc::saturated_cast<uint16_t>(time_ms - base_ms);
}

uint16_t VideoSendTiming::GetDeltaCappedMs(TimeDelta delta) {
  if (delta < TimeDelta::Zero()) {
    RTC_DLOG(LS_ERROR) << "Delta " << delta.ms()
                       << "ms expected to be positive";
  }
  return rtc::saturated_cast<uint16_t>(delta.ms());
}

TimingFrameInfo::TimingFrameInfo()
    : rtp_timestamp(0),
      capture_time_ms(-1),
      encode_start_ms(-1),
      encode_finish_ms(-1),
      packetization_finish_ms(-1),
      pacer_exit_ms(-1),
      network_timestamp_ms(-1),
      network2_timestamp_ms(-1),
      receive_start_ms(-1),
      receive_finish_ms(-1),
      decode_start_ms(-1),
      decode_finish_ms(-1),
      render_time_ms(-1),
      flags(VideoSendTiming::kNotTriggered) {}

int64_t TimingFrameInfo::EndToEndDelay() const {
  return capture_time_ms >= 0 ? decode_finish_ms - capture_time_ms : -1;
}

bool TimingFrameInfo::IsLongerThan(const TimingFrameInfo& other) const {
  int64_t other_delay = other.EndToEndDelay();
  return other_delay == -1 || EndToEndDelay() > other_delay;
}

bool TimingFrameInfo::operator<(const TimingFrameInfo& other) const {
  return other.IsLongerThan(*this);
}

bool TimingFrameInfo::operator<=(const TimingFrameInfo& other) const {
  return !IsLongerThan(other);
}

bool TimingFrameInfo::IsOutlier() const {
  return !IsInvalid() && (flags & VideoSendTiming::kTriggeredBySize);
}

bool TimingFrameInfo::IsTimerTriggered() const {
  return !IsInvalid() && (flags & VideoSendTiming::kTriggeredByTimer);
}

bool TimingFrameInfo::IsInvalid() const {
  return flags == VideoSendTiming::kInvalid;
}

std::string TimingFrameInfo::ToString() const {
  if (IsInvalid()) {
    return "";
  }

  char buf[1024];
  rtc::SimpleStringBuilder sb(buf);

  sb << rtp_timestamp << ',' << capture_time_ms << ',' << encode_start_ms << ','
     << encode_finish_ms << ',' << packetization_finish_ms << ','
     << pacer_exit_ms << ',' << network_timestamp_ms << ','
     << network2_timestamp_ms << ',' << receive_start_ms << ','
     << receive_finish_ms << ',' << decode_start_ms << ',' << decode_finish_ms
     << ',' << render_time_ms << ',' << IsOutlier() << ','
     << IsTimerTriggered();

  return sb.str();
}

VideoPlayoutDelay::VideoPlayoutDelay(TimeDelta min, TimeDelta max)
    : min_(std::clamp(min, TimeDelta::Zero(), kMax)),
      max_(std::clamp(max, min_, kMax)) {
  if (!(TimeDelta::Zero() <= min && min <= max && max <= kMax)) {
    RTC_LOG(LS_ERROR) << "Invalid video playout delay: [" << min << "," << max
                      << "]. Clamped to [" << this->min() << "," << this->max()
                      << "]";
  }
}

bool VideoPlayoutDelay::Set(TimeDelta min, TimeDelta max) {
  if (TimeDelta::Zero() <= min && min <= max && max <= kMax) {
    min_ = min;
    max_ = max;
    return true;
  }
  return false;
}

}  // namespace webrtc
