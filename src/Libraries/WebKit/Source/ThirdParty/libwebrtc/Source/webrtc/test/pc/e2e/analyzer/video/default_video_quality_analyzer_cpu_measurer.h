/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 8, 2024.
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
#ifndef TEST_PC_E2E_ANALYZER_VIDEO_DEFAULT_VIDEO_QUALITY_ANALYZER_CPU_MEASURER_H_
#define TEST_PC_E2E_ANALYZER_VIDEO_DEFAULT_VIDEO_QUALITY_ANALYZER_CPU_MEASURER_H_

#include "rtc_base/synchronization/mutex.h"

namespace webrtc {

// This class is thread safe.
class DefaultVideoQualityAnalyzerCpuMeasurer {
 public:
  double GetCpuUsagePercent() const;

  void StartMeasuringCpuProcessTime();
  void StopMeasuringCpuProcessTime();
  void StartExcludingCpuThreadTime();
  void StopExcludingCpuThreadTime();

 private:
  mutable Mutex mutex_;
  int64_t cpu_time_ RTC_GUARDED_BY(mutex_) = 0;
  int64_t wallclock_time_ RTC_GUARDED_BY(mutex_) = 0;
};

}  // namespace webrtc

#endif  // TEST_PC_E2E_ANALYZER_VIDEO_DEFAULT_VIDEO_QUALITY_ANALYZER_CPU_MEASURER_H_
