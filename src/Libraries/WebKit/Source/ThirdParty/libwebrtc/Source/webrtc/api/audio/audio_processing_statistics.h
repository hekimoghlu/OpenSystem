/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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
#ifndef API_AUDIO_AUDIO_PROCESSING_STATISTICS_H_
#define API_AUDIO_AUDIO_PROCESSING_STATISTICS_H_

#include <stdint.h>

#include <optional>

#include "rtc_base/system/rtc_export.h"

namespace webrtc {
// This version of the stats uses Optionals, it will replace the regular
// AudioProcessingStatistics struct.
struct RTC_EXPORT AudioProcessingStats {
  AudioProcessingStats();
  AudioProcessingStats(const AudioProcessingStats& other);
  ~AudioProcessingStats();

  // Deprecated.
  // TODO(bugs.webrtc.org/11226): Remove.
  // True if voice is detected in the last capture frame, after processing.
  // It is conservative in flagging audio as speech, with low likelihood of
  // incorrectly flagging a frame as voice.
  // Only reported if voice detection is enabled in AudioProcessing::Config.
  std::optional<bool> voice_detected;

  // AEC Statistics.
  // ERL = 10log_10(P_far / P_echo)
  std::optional<double> echo_return_loss;
  // ERLE = 10log_10(P_echo / P_out)
  std::optional<double> echo_return_loss_enhancement;
  // Fraction of time that the AEC linear filter is divergent, in a 1-second
  // non-overlapped aggregation window.
  std::optional<double> divergent_filter_fraction;

  // The delay metrics consists of the delay median and standard deviation. It
  // also consists of the fraction of delay estimates that can make the echo
  // cancellation perform poorly. The values are aggregated until the first
  // call to `GetStatistics()` and afterwards aggregated and updated every
  // second. Note that if there are several clients pulling metrics from
  // `GetStatistics()` during a session the first call from any of them will
  // change to one second aggregation window for all.
  std::optional<int32_t> delay_median_ms;
  std::optional<int32_t> delay_standard_deviation_ms;

  // Residual echo detector likelihood.
  std::optional<double> residual_echo_likelihood;
  // Maximum residual echo likelihood from the last time period.
  std::optional<double> residual_echo_likelihood_recent_max;

  // The instantaneous delay estimate produced in the AEC. The unit is in
  // milliseconds and the value is the instantaneous value at the time of the
  // call to `GetStatistics()`.
  std::optional<int32_t> delay_ms;
};

}  // namespace webrtc

#endif  // API_AUDIO_AUDIO_PROCESSING_STATISTICS_H_
