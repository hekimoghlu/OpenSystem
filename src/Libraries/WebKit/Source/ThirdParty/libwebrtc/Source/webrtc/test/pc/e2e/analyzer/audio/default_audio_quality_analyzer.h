/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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
#ifndef TEST_PC_E2E_ANALYZER_AUDIO_DEFAULT_AUDIO_QUALITY_ANALYZER_H_
#define TEST_PC_E2E_ANALYZER_AUDIO_DEFAULT_AUDIO_QUALITY_ANALYZER_H_

#include <map>
#include <string>

#include "absl/strings/string_view.h"
#include "api/numerics/samples_stats_counter.h"
#include "api/test/audio_quality_analyzer_interface.h"
#include "api/test/metrics/metrics_logger.h"
#include "api/test/track_id_stream_info_map.h"
#include "api/units/time_delta.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {
namespace webrtc_pc_e2e {

struct AudioStreamStats {
  SamplesStatsCounter expand_rate;
  SamplesStatsCounter accelerate_rate;
  SamplesStatsCounter preemptive_rate;
  SamplesStatsCounter speech_expand_rate;
  SamplesStatsCounter average_jitter_buffer_delay_ms;
  SamplesStatsCounter preferred_buffer_size_ms;
  SamplesStatsCounter energy;
};

class DefaultAudioQualityAnalyzer : public AudioQualityAnalyzerInterface {
 public:
  explicit DefaultAudioQualityAnalyzer(
      test::MetricsLogger* const metrics_logger);

  void Start(std::string test_case_name,
             TrackIdStreamInfoMap* analyzer_helper) override;
  void OnStatsReports(
      absl::string_view pc_label,
      const rtc::scoped_refptr<const RTCStatsReport>& report) override;
  void Stop() override;

  // Returns audio quality stats per stream label.
  std::map<std::string, AudioStreamStats> GetAudioStreamsStats() const;

 private:
  struct StatsSample {
    uint64_t total_samples_received = 0;
    uint64_t concealed_samples = 0;
    uint64_t removed_samples_for_acceleration = 0;
    uint64_t inserted_samples_for_deceleration = 0;
    uint64_t silent_concealed_samples = 0;
    TimeDelta jitter_buffer_delay = TimeDelta::Zero();
    TimeDelta jitter_buffer_target_delay = TimeDelta::Zero();
    uint64_t jitter_buffer_emitted_count = 0;
    double total_samples_duration = 0.0;
    double total_audio_energy = 0.0;
  };

  std::string GetTestCaseName(const std::string& stream_label) const;

  test::MetricsLogger* const metrics_logger_;

  std::string test_case_name_;
  TrackIdStreamInfoMap* analyzer_helper_;

  mutable Mutex lock_;
  std::map<std::string, AudioStreamStats> streams_stats_ RTC_GUARDED_BY(lock_);
  std::map<std::string, TrackIdStreamInfoMap::StreamInfo> stream_info_
      RTC_GUARDED_BY(lock_);
  std::map<std::string, StatsSample> last_stats_sample_ RTC_GUARDED_BY(lock_);
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_ANALYZER_AUDIO_DEFAULT_AUDIO_QUALITY_ANALYZER_H_
