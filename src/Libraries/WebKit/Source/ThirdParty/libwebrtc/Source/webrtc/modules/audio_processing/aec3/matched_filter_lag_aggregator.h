/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_MATCHED_FILTER_LAG_AGGREGATOR_H_
#define MODULES_AUDIO_PROCESSING_AEC3_MATCHED_FILTER_LAG_AGGREGATOR_H_

#include <optional>
#include <vector>

#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/delay_estimate.h"
#include "modules/audio_processing/aec3/matched_filter.h"

namespace webrtc {

class ApmDataDumper;

// Aggregates lag estimates produced by the MatchedFilter class into a single
// reliable combined lag estimate.
class MatchedFilterLagAggregator {
 public:
  MatchedFilterLagAggregator(ApmDataDumper* data_dumper,
                             size_t max_filter_lag,
                             const EchoCanceller3Config::Delay& delay_config);

  MatchedFilterLagAggregator() = delete;
  MatchedFilterLagAggregator(const MatchedFilterLagAggregator&) = delete;
  MatchedFilterLagAggregator& operator=(const MatchedFilterLagAggregator&) =
      delete;

  ~MatchedFilterLagAggregator();

  // Resets the aggregator.
  void Reset(bool hard_reset);

  // Aggregates the provided lag estimates.
  std::optional<DelayEstimate> Aggregate(
      const std::optional<const MatchedFilter::LagEstimate>& lag_estimate);

  // Returns whether a reliable delay estimate has been found.
  bool ReliableDelayFound() const { return significant_candidate_found_; }

  // Returns the delay candidate that is computed by looking at the highest peak
  // on the matched filters.
  int GetDelayAtHighestPeak() const {
    return highest_peak_aggregator_.candidate();
  }

 private:
  class PreEchoLagAggregator {
   public:
    PreEchoLagAggregator(size_t max_filter_lag, size_t down_sampling_factor);
    void Reset();
    void Aggregate(int pre_echo_lag);
    int pre_echo_candidate() const { return pre_echo_candidate_; }
    void Dump(ApmDataDumper* const data_dumper);

   private:
    const int block_size_log2_;
    std::array<int, 250> histogram_data_;
    std::vector<int> histogram_;
    int histogram_data_index_ = 0;
    int pre_echo_candidate_ = 0;
    int number_updates_ = 0;
  };

  class HighestPeakAggregator {
   public:
    explicit HighestPeakAggregator(size_t max_filter_lag);
    void Reset();
    void Aggregate(int lag);
    int candidate() const { return candidate_; }
    rtc::ArrayView<const int> histogram() const { return histogram_; }

   private:
    std::vector<int> histogram_;
    std::array<int, 250> histogram_data_;
    int histogram_data_index_ = 0;
    int candidate_ = -1;
  };

  ApmDataDumper* const data_dumper_;
  bool significant_candidate_found_ = false;
  const EchoCanceller3Config::Delay::DelaySelectionThresholds thresholds_;
  const int headroom_;
  HighestPeakAggregator highest_peak_aggregator_;
  std::unique_ptr<PreEchoLagAggregator> pre_echo_lag_aggregator_;
};
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_MATCHED_FILTER_LAG_AGGREGATOR_H_
