/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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
#ifndef RTC_BASE_NUMERICS_SAMPLE_STATS_H_
#define RTC_BASE_NUMERICS_SAMPLE_STATS_H_

#include "api/numerics/samples_stats_counter.h"
#include "api/units/data_rate.h"
#include "api/units/time_delta.h"

namespace webrtc {
template <typename T>
class SampleStats;

// TODO(srte): Merge this implementation with SamplesStatsCounter.
template <>
class SampleStats<double> : public SamplesStatsCounter {
 public:
  double Max();
  double Mean();
  double Median();
  double Quantile(double quantile);
  double Min();
  double Variance();
  double StandardDeviation();
  int Count();
};

template <>
class SampleStats<TimeDelta> {
 public:
  void AddSample(TimeDelta delta);
  void AddSampleMs(double delta_ms);
  void AddSamples(const SampleStats<TimeDelta>& other);
  bool IsEmpty();
  TimeDelta Max();
  TimeDelta Mean();
  TimeDelta Median();
  TimeDelta Quantile(double quantile);
  TimeDelta Min();
  TimeDelta Variance();
  TimeDelta StandardDeviation();
  int Count();

 private:
  SampleStats<double> stats_;
};

template <>
class SampleStats<DataRate> {
 public:
  void AddSample(DataRate rate);
  void AddSampleBps(double rate_bps);
  void AddSamples(const SampleStats<DataRate>& other);
  bool IsEmpty();
  DataRate Max();
  DataRate Mean();
  DataRate Median();
  DataRate Quantile(double quantile);
  DataRate Min();
  DataRate Variance();
  DataRate StandardDeviation();
  int Count();

 private:
  SampleStats<double> stats_;
};
}  // namespace webrtc

#endif  // RTC_BASE_NUMERICS_SAMPLE_STATS_H_
