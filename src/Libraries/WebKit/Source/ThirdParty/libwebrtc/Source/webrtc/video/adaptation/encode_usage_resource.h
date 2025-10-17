/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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
#ifndef VIDEO_ADAPTATION_ENCODE_USAGE_RESOURCE_H_
#define VIDEO_ADAPTATION_ENCODE_USAGE_RESOURCE_H_

#include <memory>
#include <optional>
#include <string>

#include "api/scoped_refptr.h"
#include "api/video/video_adaptation_reason.h"
#include "video/adaptation/overuse_frame_detector.h"
#include "video/adaptation/video_stream_encoder_resource.h"

namespace webrtc {

// Handles interaction with the OveruseDetector.
// TODO(hbos): Add unittests specific to this class, it is currently only tested
// indirectly by usage in the ResourceAdaptationProcessor (which is only tested
// because of its usage in VideoStreamEncoder); all tests are currently in
// video_stream_encoder_unittest.cc.
class EncodeUsageResource : public VideoStreamEncoderResource,
                            public OveruseFrameDetectorObserverInterface {
 public:
  static rtc::scoped_refptr<EncodeUsageResource> Create(
      std::unique_ptr<OveruseFrameDetector> overuse_detector);

  explicit EncodeUsageResource(
      std::unique_ptr<OveruseFrameDetector> overuse_detector);
  ~EncodeUsageResource() override;

  bool is_started() const;

  void StartCheckForOveruse(CpuOveruseOptions options);
  void StopCheckForOveruse();

  void SetTargetFrameRate(std::optional<double> target_frame_rate);
  void OnEncodeStarted(const VideoFrame& cropped_frame,
                       int64_t time_when_first_seen_us);
  void OnEncodeCompleted(uint32_t timestamp,
                         int64_t time_sent_in_us,
                         int64_t capture_time_us,
                         std::optional<int> encode_duration_us);

  // OveruseFrameDetectorObserverInterface implementation.
  void AdaptUp() override;
  void AdaptDown() override;

 private:
  int TargetFrameRateAsInt();

  const std::unique_ptr<OveruseFrameDetector> overuse_detector_
      RTC_GUARDED_BY(encoder_queue());
  bool is_started_ RTC_GUARDED_BY(encoder_queue());
  std::optional<double> target_frame_rate_ RTC_GUARDED_BY(encoder_queue());
};

}  // namespace webrtc

#endif  // VIDEO_ADAPTATION_ENCODE_USAGE_RESOURCE_H_
