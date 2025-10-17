/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_PERFORMANCE_TIMER_H_
#define MODULES_AUDIO_PROCESSING_TEST_PERFORMANCE_TIMER_H_

#include <optional>
#include <vector>

#include "system_wrappers/include/clock.h"

namespace webrtc {
namespace test {

class PerformanceTimer {
 public:
  explicit PerformanceTimer(int num_frames_to_process);
  ~PerformanceTimer();

  void StartTimer();
  void StopTimer();

  double GetDurationAverage() const;
  double GetDurationStandardDeviation() const;

  // These methods are the same as those above, but they ignore the first
  // `number_of_warmup_samples` measurements.
  double GetDurationAverage(size_t number_of_warmup_samples) const;
  double GetDurationStandardDeviation(size_t number_of_warmup_samples) const;

 private:
  webrtc::Clock* clock_;
  std::optional<int64_t> start_timestamp_us_;
  std::vector<int64_t> timestamps_us_;
};

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_PERFORMANCE_TIMER_H_
