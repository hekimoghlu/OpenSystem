/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
#ifndef TEST_PC_E2E_CROSS_MEDIA_METRICS_REPORTER_H_
#define TEST_PC_E2E_CROSS_MEDIA_METRICS_REPORTER_H_

#include <map>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "api/numerics/samples_stats_counter.h"
#include "api/test/metrics/metrics_logger.h"
#include "api/test/peerconnection_quality_test_fixture.h"
#include "api/test/track_id_stream_info_map.h"
#include "api/units/timestamp.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {
namespace webrtc_pc_e2e {

class CrossMediaMetricsReporter
    : public PeerConnectionE2EQualityTestFixture::QualityMetricsReporter {
 public:
  explicit CrossMediaMetricsReporter(test::MetricsLogger* metrics_logger);
  ~CrossMediaMetricsReporter() override = default;

  void Start(absl::string_view test_case_name,
             const TrackIdStreamInfoMap* reporter_helper) override;
  void OnStatsReports(
      absl::string_view pc_label,
      const rtc::scoped_refptr<const RTCStatsReport>& report) override;
  void StopAndReportResults() override;

 private:
  struct StatsInfo {
    SamplesStatsCounter audio_ahead_ms;
    SamplesStatsCounter video_ahead_ms;

    TrackIdStreamInfoMap::StreamInfo audio_stream_info;
    TrackIdStreamInfoMap::StreamInfo video_stream_info;
    std::string audio_stream_label;
    std::string video_stream_label;
  };

  std::string GetTestCaseName(const std::string& stream_label,
                              const std::string& sync_group) const;

  test::MetricsLogger* const metrics_logger_;

  std::string test_case_name_;
  const TrackIdStreamInfoMap* reporter_helper_;

  Mutex mutex_;
  std::map<std::string, StatsInfo> stats_info_ RTC_GUARDED_BY(mutex_);
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_CROSS_MEDIA_METRICS_REPORTER_H_
