/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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
#ifndef MODULES_VIDEO_CODING_CODECS_TEST_VIDEOCODEC_TEST_FIXTURE_IMPL_H_
#define MODULES_VIDEO_CODING_CODECS_TEST_VIDEOCODEC_TEST_FIXTURE_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#include "api/environment/environment.h"
#include "api/test/videocodec_test_fixture.h"
#include "api/video_codecs/video_decoder_factory.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "common_video/h264/h264_common.h"
#include "modules/video_coding/codecs/test/videocodec_test_stats_impl.h"
#include "modules/video_coding/codecs/test/videoprocessor.h"
#include "modules/video_coding/utility/ivf_file_writer.h"
#include "rtc_base/task_queue_for_test.h"
#include "test/testsupport/frame_reader.h"
#include "test/testsupport/frame_writer.h"

namespace webrtc {
namespace test {

// Integration test for video processor. It does rate control and frame quality
// analysis using frame statistics collected by video processor and logs the
// results. If thresholds are specified it checks that corresponding metrics
// are in desirable range.
class VideoCodecTestFixtureImpl : public VideoCodecTestFixture {
  // Verifies that all H.264 keyframes contain SPS/PPS/IDR NALUs.
 public:
  class H264KeyframeChecker : public EncodedFrameChecker {
   public:
    void CheckEncodedFrame(webrtc::VideoCodecType codec,
                           const EncodedImage& encoded_frame) const override;
  };

  explicit VideoCodecTestFixtureImpl(Config config);
  VideoCodecTestFixtureImpl(
      Config config,
      std::unique_ptr<VideoDecoderFactory> decoder_factory,
      std::unique_ptr<VideoEncoderFactory> encoder_factory);
  ~VideoCodecTestFixtureImpl() override;

  void RunTest(const std::vector<RateProfile>& rate_profiles,
               const std::vector<RateControlThresholds>* rc_thresholds,
               const std::vector<QualityThresholds>* quality_thresholds,
               const BitstreamThresholds* bs_thresholds) override;

  VideoCodecTestStats& GetStats() override;

 private:
  class CpuProcessTime;

  bool CreateEncoderAndDecoder();
  void DestroyEncoderAndDecoder();
  bool SetUpAndInitObjects(TaskQueueForTest* task_queue,
                           size_t initial_bitrate_kbps,
                           double initial_framerate_fps);
  void ReleaseAndCloseObjects(TaskQueueForTest* task_queue);

  void ProcessAllFrames(TaskQueueForTest* task_queue,
                        const std::vector<RateProfile>& rate_profiles);
  void AnalyzeAllFrames(
      const std::vector<RateProfile>& rate_profiles,
      const std::vector<RateControlThresholds>* rc_thresholds,
      const std::vector<QualityThresholds>* quality_thresholds,
      const BitstreamThresholds* bs_thresholds);

  void VerifyVideoStatistic(
      const VideoCodecTestStats::VideoStatistics& video_stat,
      const RateControlThresholds* rc_thresholds,
      const QualityThresholds* quality_thresholds,
      const BitstreamThresholds* bs_thresholds,
      size_t target_bitrate_kbps,
      double input_framerate_fps);

  std::string GetCodecName(TaskQueueForTest* task_queue, bool is_encoder) const;
  void PrintSettings(TaskQueueForTest* task_queue) const;

  // Codecs.
  const std::unique_ptr<VideoEncoderFactory> encoder_factory_;
  std::unique_ptr<VideoEncoder> encoder_;
  const std::unique_ptr<VideoDecoderFactory> decoder_factory_;
  VideoProcessor::VideoDecoderList decoders_;

  // Helper objects.
  const Environment env_;
  Config config_;
  VideoCodecTestStatsImpl stats_;
  std::unique_ptr<FrameReader> source_frame_reader_;
  VideoProcessor::IvfFileWriterMap encoded_frame_writers_;
  VideoProcessor::FrameWriterList decoded_frame_writers_;
  std::unique_ptr<VideoProcessor> processor_;
  std::unique_ptr<CpuProcessTime> cpu_process_time_;
};

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_CODECS_TEST_VIDEOCODEC_TEST_FIXTURE_IMPL_H_
