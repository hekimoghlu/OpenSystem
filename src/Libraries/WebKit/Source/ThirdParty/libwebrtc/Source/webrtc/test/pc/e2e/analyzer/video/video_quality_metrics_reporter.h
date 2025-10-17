/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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
#ifndef TEST_PC_E2E_ANALYZER_VIDEO_VIDEO_QUALITY_METRICS_REPORTER_H_
#define TEST_PC_E2E_ANALYZER_VIDEO_VIDEO_QUALITY_METRICS_REPORTER_H_

#include <map>
#include <string>

#include "absl/strings/string_view.h"
#include "api/numerics/samples_stats_counter.h"
#include "api/test/metrics/metrics_logger.h"
#include "api/test/peerconnection_quality_test_fixture.h"
#include "api/test/track_id_stream_info_map.h"
#include "api/units/data_size.h"
#include "api/units/timestamp.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {
namespace webrtc_pc_e2e {

struct VideoBweStats {
  SamplesStatsCounter available_send_bandwidth;
  SamplesStatsCounter transmission_bitrate;
  SamplesStatsCounter retransmission_bitrate;
};

class VideoQualityMetricsReporter
    : public PeerConnectionE2EQualityTestFixture::QualityMetricsReporter {
 public:
  VideoQualityMetricsReporter(Clock* const clock,
                              test::MetricsLogger* const metrics_logger);
  ~VideoQualityMetricsReporter() override = default;

  void Start(absl::string_view test_case_name,
             const TrackIdStreamInfoMap* reporter_helper) override;
  void OnStatsReports(
      absl::string_view pc_label,
      const rtc::scoped_refptr<const RTCStatsReport>& report) override;
  void StopAndReportResults() override;

 private:
  struct StatsSample {
    DataSize bytes_sent = DataSize::Zero();
    DataSize header_bytes_sent = DataSize::Zero();
    DataSize retransmitted_bytes_sent = DataSize::Zero();

    Timestamp sample_time = Timestamp::Zero();
  };

  std::string GetTestCaseName(const std::string& peer_name) const;
  void ReportVideoBweResults(const std::string& peer_name,
                             const VideoBweStats& video_bwe_stats);
  Timestamp Now() const { return clock_->CurrentTime(); }

  Clock* const clock_;
  test::MetricsLogger* const metrics_logger_;

  std::string test_case_name_;
  std::optional<Timestamp> start_time_;

  Mutex video_bwe_stats_lock_;
  // Map between a peer connection label (provided by the framework) and
  // its video BWE stats.
  std::map<std::string, VideoBweStats> video_bwe_stats_
      RTC_GUARDED_BY(video_bwe_stats_lock_);
  std::map<std::string, StatsSample> last_stats_sample_
      RTC_GUARDED_BY(video_bwe_stats_lock_);
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_ANALYZER_VIDEO_VIDEO_QUALITY_METRICS_REPORTER_H_
