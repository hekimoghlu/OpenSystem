/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 15, 2022.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_STATS_GETTER_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_STATS_GETTER_H_

#include <memory>
#include <string>
#include <vector>

#include "modules/audio_coding/neteq/tools/neteq_delay_analyzer.h"
#include "modules/audio_coding/neteq/tools/neteq_test.h"

namespace webrtc {
namespace test {

class NetEqStatsGetter : public NetEqGetAudioCallback {
 public:
  // This struct is a replica of webrtc::NetEqNetworkStatistics, but with all
  // values stored in double precision.
  struct Stats {
    double current_buffer_size_ms = 0.0;
    double preferred_buffer_size_ms = 0.0;
    double jitter_peaks_found = 0.0;
    double packet_loss_rate = 0.0;
    double expand_rate = 0.0;
    double speech_expand_rate = 0.0;
    double preemptive_rate = 0.0;
    double accelerate_rate = 0.0;
    double secondary_decoded_rate = 0.0;
    double secondary_discarded_rate = 0.0;
    double clockdrift_ppm = 0.0;
    double added_zero_samples = 0.0;
    double mean_waiting_time_ms = 0.0;
    double median_waiting_time_ms = 0.0;
    double min_waiting_time_ms = 0.0;
    double max_waiting_time_ms = 0.0;
  };

  struct ConcealmentEvent {
    uint64_t duration_ms;
    size_t concealment_event_number;
    int64_t time_from_previous_event_end_ms;
    std::string ToString() const;
  };

  // Takes a pointer to another callback object, which will be invoked after
  // this object finishes. This does not transfer ownership, and null is a
  // valid value.
  explicit NetEqStatsGetter(std::unique_ptr<NetEqDelayAnalyzer> delay_analyzer);

  void set_stats_query_interval_ms(int64_t stats_query_interval_ms) {
    stats_query_interval_ms_ = stats_query_interval_ms;
  }

  void BeforeGetAudio(NetEq* neteq) override;

  void AfterGetAudio(int64_t time_now_ms,
                     const AudioFrame& audio_frame,
                     bool muted,
                     NetEq* neteq) override;

  double AverageSpeechExpandRate() const;

  NetEqDelayAnalyzer* delay_analyzer() const { return delay_analyzer_.get(); }

  const std::vector<ConcealmentEvent>& concealment_events() const {
    // Do not account for the last concealment event to avoid potential end
    // call skewing.
    return concealment_events_;
  }

  const std::vector<std::pair<int64_t, NetEqNetworkStatistics>>* stats() const {
    return &stats_;
  }

  const std::vector<std::pair<int64_t, NetEqLifetimeStatistics>>*
  lifetime_stats() const {
    return &lifetime_stats_;
  }

  Stats AverageStats() const;

 private:
  std::unique_ptr<NetEqDelayAnalyzer> delay_analyzer_;
  int64_t stats_query_interval_ms_ = 1000;
  int64_t last_stats_query_time_ms_ = 0;
  std::vector<std::pair<int64_t, NetEqNetworkStatistics>> stats_;
  std::vector<std::pair<int64_t, NetEqLifetimeStatistics>> lifetime_stats_;
  size_t current_concealment_event_ = 1;
  uint64_t voice_concealed_samples_until_last_event_ = 0;
  std::vector<ConcealmentEvent> concealment_events_;
  int64_t last_event_end_time_ms_ = 0;
};

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_STATS_GETTER_H_
