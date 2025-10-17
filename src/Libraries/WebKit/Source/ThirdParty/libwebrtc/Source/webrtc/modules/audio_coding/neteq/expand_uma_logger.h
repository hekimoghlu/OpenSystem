/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 14, 2024.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_EXPAND_UMA_LOGGER_H_
#define MODULES_AUDIO_CODING_NETEQ_EXPAND_UMA_LOGGER_H_

#include <stdint.h>

#include <memory>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "api/neteq/tick_timer.h"

namespace webrtc {

// This class is used to periodically log values to a UMA histogram. The caller
// is expected to update this class with an incremental sample counter which
// counts expand samples. At the end of each logging period, the class will
// calculate the fraction of samples that were expand samples during that period
// and report that in percent. The logging period must be strictly positive.
// Does not take ownership of tick_timer and the pointer must refer to a valid
// object that outlives the one constructed.
class ExpandUmaLogger {
 public:
  ExpandUmaLogger(absl::string_view uma_name,
                  int logging_period_s,
                  const TickTimer* tick_timer);

  ~ExpandUmaLogger();

  ExpandUmaLogger(const ExpandUmaLogger&) = delete;
  ExpandUmaLogger& operator=(const ExpandUmaLogger&) = delete;

  // In this call, value should be an incremental sample counter. The sample
  // rate must be strictly positive.
  void UpdateSampleCounter(uint64_t value, int sample_rate_hz);

 private:
  const std::string uma_name_;
  const int logging_period_s_;
  const TickTimer& tick_timer_;
  std::unique_ptr<TickTimer::Countdown> timer_;
  std::optional<uint64_t> last_logged_value_;
  uint64_t last_value_ = 0;
  int sample_rate_hz_ = 0;
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_EXPAND_UMA_LOGGER_H_
