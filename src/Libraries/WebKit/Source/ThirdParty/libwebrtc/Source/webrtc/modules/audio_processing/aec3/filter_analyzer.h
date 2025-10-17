/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_FILTER_ANALYZER_H_
#define MODULES_AUDIO_PROCESSING_AEC3_FILTER_ANALYZER_H_

#include <stddef.h>

#include <array>
#include <atomic>
#include <memory>
#include <vector>

#include "api/array_view.h"
#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/block.h"

namespace webrtc {

class ApmDataDumper;
class RenderBuffer;

// Class for analyzing the properties of an adaptive filter.
class FilterAnalyzer {
 public:
  FilterAnalyzer(const EchoCanceller3Config& config,
                 size_t num_capture_channels);
  ~FilterAnalyzer();

  FilterAnalyzer(const FilterAnalyzer&) = delete;
  FilterAnalyzer& operator=(const FilterAnalyzer&) = delete;

  // Resets the analysis.
  void Reset();

  // Updates the estimates with new input data.
  void Update(rtc::ArrayView<const std::vector<float>> filters_time_domain,
              const RenderBuffer& render_buffer,
              bool* any_filter_consistent,
              float* max_echo_path_gain);

  // Returns the delay in blocks for each filter.
  rtc::ArrayView<const int> FilterDelaysBlocks() const {
    return filter_delays_blocks_;
  }

  // Returns the minimum delay of all filters in terms of blocks.
  int MinFilterDelayBlocks() const { return min_filter_delay_blocks_; }

  // Returns the number of blocks for the current used filter.
  int FilterLengthBlocks() const {
    return filter_analysis_states_[0].filter_length_blocks;
  }

  // Returns the preprocessed filter.
  rtc::ArrayView<const std::vector<float>> GetAdjustedFilters() const {
    return h_highpass_;
  }

  // Public for testing purposes only.
  void SetRegionToAnalyze(size_t filter_size);

 private:
  struct FilterAnalysisState;

  void AnalyzeRegion(
      rtc::ArrayView<const std::vector<float>> filters_time_domain,
      const RenderBuffer& render_buffer);

  void UpdateFilterGain(rtc::ArrayView<const float> filters_time_domain,
                        FilterAnalysisState* st);
  void PreProcessFilters(
      rtc::ArrayView<const std::vector<float>> filters_time_domain);

  void ResetRegion();

  struct FilterRegion {
    size_t start_sample_;
    size_t end_sample_;
  };

  // This class checks whether the shape of the impulse response has been
  // consistent over time.
  class ConsistentFilterDetector {
   public:
    explicit ConsistentFilterDetector(const EchoCanceller3Config& config);
    void Reset();
    bool Detect(rtc::ArrayView<const float> filter_to_analyze,
                const FilterRegion& region,
                const Block& x_block,
                size_t peak_index,
                int delay_blocks);

   private:
    bool significant_peak_;
    float filter_floor_accum_;
    float filter_secondary_peak_;
    size_t filter_floor_low_limit_;
    size_t filter_floor_high_limit_;
    const float active_render_threshold_;
    size_t consistent_estimate_counter_ = 0;
    int consistent_delay_reference_ = -10;
  };

  struct FilterAnalysisState {
    explicit FilterAnalysisState(const EchoCanceller3Config& config)
        : filter_length_blocks(config.filter.refined_initial.length_blocks),
          consistent_filter_detector(config) {
      Reset(config.ep_strength.default_gain);
    }

    void Reset(float default_gain) {
      peak_index = 0;
      gain = default_gain;
      consistent_filter_detector.Reset();
    }

    float gain;
    size_t peak_index;
    int filter_length_blocks;
    bool consistent_estimate = false;
    ConsistentFilterDetector consistent_filter_detector;
  };

  static std::atomic<int> instance_count_;
  std::unique_ptr<ApmDataDumper> data_dumper_;
  const bool bounded_erl_;
  const float default_gain_;
  std::vector<std::vector<float>> h_highpass_;

  size_t blocks_since_reset_ = 0;
  FilterRegion region_;

  std::vector<FilterAnalysisState> filter_analysis_states_;
  std::vector<int> filter_delays_blocks_;

  int min_filter_delay_blocks_ = 0;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_FILTER_ANALYZER_H_
