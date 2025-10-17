/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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
#include "test/pc/e2e/analyzer/video/default_video_quality_analyzer_cpu_measurer.h"

#include "rtc_base/cpu_time.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/system_time.h"

namespace webrtc {

void DefaultVideoQualityAnalyzerCpuMeasurer::StartMeasuringCpuProcessTime() {
  MutexLock lock(&mutex_);
  cpu_time_ -= rtc::GetProcessCpuTimeNanos();
  wallclock_time_ -= rtc::SystemTimeNanos();
}

void DefaultVideoQualityAnalyzerCpuMeasurer::StopMeasuringCpuProcessTime() {
  MutexLock lock(&mutex_);
  cpu_time_ += rtc::GetProcessCpuTimeNanos();
  wallclock_time_ += rtc::SystemTimeNanos();
}

void DefaultVideoQualityAnalyzerCpuMeasurer::StartExcludingCpuThreadTime() {
  MutexLock lock(&mutex_);
  cpu_time_ += rtc::GetThreadCpuTimeNanos();
}

void DefaultVideoQualityAnalyzerCpuMeasurer::StopExcludingCpuThreadTime() {
  MutexLock lock(&mutex_);
  cpu_time_ -= rtc::GetThreadCpuTimeNanos();
}

double DefaultVideoQualityAnalyzerCpuMeasurer::GetCpuUsagePercent() const {
  MutexLock lock(&mutex_);
  return static_cast<double>(cpu_time_) / wallclock_time_ * 100.0;
}

}  // namespace webrtc
