/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 16, 2024.
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
#include "pc/jitter_buffer_delay.h"

#include "api/sequence_checker.h"
#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_conversions.h"
#include "rtc_base/numerics/safe_minmax.h"

namespace {
constexpr int kDefaultDelay = 0;
constexpr int kMaximumDelayMs = 10000;
}  // namespace

namespace webrtc {

void JitterBufferDelay::Set(std::optional<double> delay_seconds) {
  RTC_DCHECK_RUN_ON(&worker_thread_checker_);
  cached_delay_seconds_ = delay_seconds;
}

int JitterBufferDelay::GetMs() const {
  RTC_DCHECK_RUN_ON(&worker_thread_checker_);
  return rtc::SafeClamp(
      rtc::saturated_cast<int>(cached_delay_seconds_.value_or(kDefaultDelay) *
                               1000),
      0, kMaximumDelayMs);
}

}  // namespace webrtc
