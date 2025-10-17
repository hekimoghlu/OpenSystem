/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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
#include "modules/audio_coding/neteq/expand_uma_logger.h"

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"
#include "system_wrappers/include/metrics.h"

namespace webrtc {
namespace {
std::unique_ptr<TickTimer::Countdown> GetNewCountdown(
    const TickTimer& tick_timer,
    int logging_period_s) {
  return tick_timer.GetNewCountdown((logging_period_s * 1000) /
                                    tick_timer.ms_per_tick());
}
}  // namespace

ExpandUmaLogger::ExpandUmaLogger(absl::string_view uma_name,
                                 int logging_period_s,
                                 const TickTimer* tick_timer)
    : uma_name_(uma_name),
      logging_period_s_(logging_period_s),
      tick_timer_(*tick_timer),
      timer_(GetNewCountdown(tick_timer_, logging_period_s_)) {
  RTC_DCHECK(tick_timer);
  RTC_DCHECK_GT(logging_period_s_, 0);
}

ExpandUmaLogger::~ExpandUmaLogger() = default;

void ExpandUmaLogger::UpdateSampleCounter(uint64_t samples,
                                          int sample_rate_hz) {
  if ((last_logged_value_ && *last_logged_value_ > samples) ||
      sample_rate_hz_ != sample_rate_hz) {
    // Sanity checks. The incremental counter moved backwards, or sample rate
    // changed.
    last_logged_value_.reset();
  }
  last_value_ = samples;
  sample_rate_hz_ = sample_rate_hz;
  if (!last_logged_value_) {
    last_logged_value_ = std::optional<uint64_t>(samples);
  }

  if (!timer_->Finished()) {
    // Not yet time to log.
    return;
  }

  RTC_DCHECK(last_logged_value_);
  RTC_DCHECK_GE(last_value_, *last_logged_value_);
  const uint64_t diff = last_value_ - *last_logged_value_;
  last_logged_value_ = std::optional<uint64_t>(last_value_);
  // Calculate rate in percent.
  RTC_DCHECK_GT(sample_rate_hz, 0);
  const int rate = (100 * diff) / (sample_rate_hz * logging_period_s_);
  RTC_DCHECK_GE(rate, 0);
  RTC_DCHECK_LE(rate, 100);
  RTC_HISTOGRAM_PERCENTAGE_SPARSE(uma_name_, rate);
  timer_ = GetNewCountdown(tick_timer_, logging_period_s_);
}

}  // namespace webrtc
