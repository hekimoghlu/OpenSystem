/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 12, 2021.
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
#ifndef TEST_SCENARIO_STATS_COLLECTION_H_
#define TEST_SCENARIO_STATS_COLLECTION_H_

#include <map>
#include <memory>
#include <optional>

#include "call/call.h"
#include "rtc_base/thread.h"
#include "test/logging/log_writer.h"
#include "test/scenario/performance_stats.h"

namespace webrtc {
namespace test {

struct VideoQualityAnalyzerConfig {
  double psnr_coverage = 1;
  rtc::Thread* thread = nullptr;
};

class VideoLayerAnalyzer {
 public:
  void HandleCapturedFrame(const VideoFramePair& sample);
  void HandleRenderedFrame(const VideoFramePair& sample);
  void HandleFramePair(VideoFramePair sample,
                       double psnr,
                       RtcEventLogOutput* writer);
  VideoQualityStats stats_;
  Timestamp last_capture_time_ = Timestamp::MinusInfinity();
  Timestamp last_render_time_ = Timestamp::MinusInfinity();
  Timestamp last_freeze_time_ = Timestamp::MinusInfinity();
  int skip_count_ = 0;
};

class VideoQualityAnalyzer {
 public:
  explicit VideoQualityAnalyzer(
      VideoQualityAnalyzerConfig config = VideoQualityAnalyzerConfig(),
      std::unique_ptr<RtcEventLogOutput> writer = nullptr);
  ~VideoQualityAnalyzer();
  void HandleFramePair(VideoFramePair sample);
  std::vector<VideoQualityStats> layer_stats() const;
  VideoQualityStats& stats();
  void PrintHeaders();
  void PrintFrameInfo(const VideoFramePair& sample);
  std::function<void(const VideoFramePair&)> Handler();

 private:
  void HandleFramePair(VideoFramePair sample, double psnr);
  const VideoQualityAnalyzerConfig config_;
  std::map<int, VideoLayerAnalyzer> layer_analyzers_;
  const std::unique_ptr<RtcEventLogOutput> writer_;
  std::optional<VideoQualityStats> cached_;
};

class CallStatsCollector {
 public:
  void AddStats(Call::Stats sample);
  CollectedCallStats& stats() { return stats_; }

 private:
  CollectedCallStats stats_;
};
class AudioReceiveStatsCollector {
 public:
  void AddStats(AudioReceiveStreamInterface::Stats sample);
  CollectedAudioReceiveStats& stats() { return stats_; }

 private:
  CollectedAudioReceiveStats stats_;
};
class VideoSendStatsCollector {
 public:
  void AddStats(VideoSendStream::Stats sample, Timestamp at_time);
  CollectedVideoSendStats& stats() { return stats_; }

 private:
  CollectedVideoSendStats stats_;
  Timestamp last_update_ = Timestamp::MinusInfinity();
  size_t last_fec_bytes_ = 0;
};
class VideoReceiveStatsCollector {
 public:
  void AddStats(VideoReceiveStreamInterface::Stats sample);
  CollectedVideoReceiveStats& stats() { return stats_; }

 private:
  CollectedVideoReceiveStats stats_;
};

struct CallStatsCollectors {
  CallStatsCollector call;
  AudioReceiveStatsCollector audio_receive;
  VideoSendStatsCollector video_send;
  VideoReceiveStatsCollector video_receive;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_SCENARIO_STATS_COLLECTION_H_
