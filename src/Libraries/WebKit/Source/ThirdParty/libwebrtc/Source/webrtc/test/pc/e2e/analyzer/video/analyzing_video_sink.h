/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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
#ifndef TEST_PC_E2E_ANALYZER_VIDEO_ANALYZING_VIDEO_SINK_H_
#define TEST_PC_E2E_ANALYZER_VIDEO_ANALYZING_VIDEO_SINK_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/numerics/samples_stats_counter.h"
#include "api/test/metrics/metrics_logger.h"
#include "api/test/pclf/media_configuration.h"
#include "api/test/video/video_frame_writer.h"
#include "api/test/video_quality_analyzer_interface.h"
#include "api/video/video_frame.h"
#include "api/video/video_sink_interface.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"
#include "system_wrappers/include/clock.h"
#include "test/pc/e2e/analyzer/video/analyzing_video_sinks_helper.h"

namespace webrtc {
namespace webrtc_pc_e2e {

// A sink to inject video quality analyzer as a sink into WebRTC.
class AnalyzingVideoSink : public rtc::VideoSinkInterface<VideoFrame> {
 public:
  struct Stats {
    // Time required to scale video frame to the requested rendered resolution.
    // Collected only for frames with ID set and iff `report_infra_stats` is
    // true.
    SamplesStatsCounter scaling_tims_ms;
    // Time required to process single video frame. Collected only for frames
    // with ID set and iff `report_infra_stats` is true.
    SamplesStatsCounter analyzing_sink_processing_time_ms;
  };

  AnalyzingVideoSink(absl::string_view peer_name,
                     Clock* clock,
                     VideoQualityAnalyzerInterface& analyzer,
                     AnalyzingVideoSinksHelper& sinks_helper,
                     const VideoSubscription& subscription,
                     bool report_infra_stats);

  // Updates subscription used by this peer to render received video.
  void UpdateSubscription(const VideoSubscription& subscription);

  void OnFrame(const VideoFrame& frame) override;

  void LogMetrics(webrtc::test::MetricsLogger& metrics_logger,
                  absl::string_view test_case_name) const;

  Stats stats() const;

 private:
  struct SinksDescriptor {
    SinksDescriptor(absl::string_view sender_peer_name,
                    const VideoResolution& resolution)
        : sender_peer_name(sender_peer_name), resolution(resolution) {}

    // Required to be able to resolve resolutions on new subscription and
    // understand if we need to recreate `video_frame_writer` and `sinks`.
    std::string sender_peer_name;
    // Resolution which was used to create `video_frame_writer` and `sinks`.
    VideoResolution resolution;

    // Is set if dumping of output video was requested;
    test::VideoFrameWriter* video_frame_writer = nullptr;
    std::vector<std::unique_ptr<rtc::VideoSinkInterface<VideoFrame>>> sinks;
  };

  // Scales video frame to `required_resolution` if necessary. Crashes if video
  // frame and `required_resolution` have different aspect ratio.
  VideoFrame ScaleVideoFrame(const VideoFrame& frame,
                             const VideoResolution& required_resolution)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  // Creates full copy of the frame to free any frame owned internal buffers
  // and passes created copy to analyzer. Uses `I420Buffer` to represent
  // frame content.
  void AnalyzeFrame(const VideoFrame& frame);
  // Populates sink for specified stream and caches them in `stream_sinks_`.
  SinksDescriptor* PopulateSinks(absl::string_view stream_label)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  const std::string peer_name_;
  const bool report_infra_stats_;
  Clock* const clock_;
  VideoQualityAnalyzerInterface* const analyzer_;
  AnalyzingVideoSinksHelper* const sinks_helper_;

  mutable Mutex mutex_;
  VideoSubscription subscription_ RTC_GUARDED_BY(mutex_);
  std::map<std::string, SinksDescriptor> stream_sinks_ RTC_GUARDED_BY(mutex_);
  Stats stats_ RTC_GUARDED_BY(mutex_);
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_ANALYZER_VIDEO_ANALYZING_VIDEO_SINK_H_
