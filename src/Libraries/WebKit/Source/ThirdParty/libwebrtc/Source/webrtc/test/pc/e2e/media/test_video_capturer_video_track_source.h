/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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
#ifndef TEST_PC_E2E_MEDIA_TEST_VIDEO_CAPTURER_VIDEO_TRACK_SOURCE_H_
#define TEST_PC_E2E_MEDIA_TEST_VIDEO_CAPTURER_VIDEO_TRACK_SOURCE_H_

#include <memory>
#include <optional>
#include <utility>

#include "api/sequence_checker.h"
#include "api/test/video/test_video_track_source.h"
#include "api/video/video_frame.h"
#include "api/video/video_source_interface.h"
#include "test/test_video_capturer.h"

namespace webrtc {
namespace webrtc_pc_e2e {

class TestVideoCapturerVideoTrackSource : public test::TestVideoTrackSource {
 public:
  TestVideoCapturerVideoTrackSource(
      std::unique_ptr<test::TestVideoCapturer> video_capturer,
      bool is_screencast,
      std::optional<std::string> stream_label = std::nullopt)
      : TestVideoTrackSource(/*remote=*/false, std::move(stream_label)),
        video_capturer_(std::move(video_capturer)),
        is_screencast_(is_screencast) {
    sequence_checker_.Detach();
  }

  TestVideoCapturerVideoTrackSource(const TestVideoCapturerVideoTrackSource&) =
      delete;
  TestVideoCapturerVideoTrackSource& operator=(
      const TestVideoCapturerVideoTrackSource&) = delete;
  TestVideoCapturerVideoTrackSource(TestVideoCapturerVideoTrackSource&&) =
      delete;
  TestVideoCapturerVideoTrackSource& operator=(
      TestVideoCapturerVideoTrackSource&&) = delete;

  ~TestVideoCapturerVideoTrackSource() = default;

  void Start() override {
    SetState(kLive);
    video_capturer_->Start();
  }

  void Stop() override {
    SetState(kMuted);
    video_capturer_->Stop();
  }

  int GetFrameWidth() const override {
    return video_capturer_->GetFrameWidth();
  }

  int GetFrameHeight() const override {
    return video_capturer_->GetFrameHeight();
  }

  bool is_screencast() const override {
    RTC_DCHECK_RUN_ON(&sequence_checker_);
    return is_screencast_;
  }

  void SetEnableAdaptation(bool enable_adaptation) {
    video_capturer_->SetEnableAdaptation(enable_adaptation);
  }

  void OnOutputFormatRequest(int width,
                             int height,
                             const std::optional<int>& max_fps) override {
    video_capturer_->OnOutputFormatRequest(width, height, max_fps);
  }

  void SetScreencast(bool is_screencast) override {
    RTC_DCHECK_RUN_ON(&sequence_checker_);
    is_screencast_ = is_screencast;
  }

 protected:
  rtc::VideoSourceInterface<VideoFrame>* source() override {
    return video_capturer_.get();
  }

 private:
  const std::unique_ptr<test::TestVideoCapturer> video_capturer_;
  SequenceChecker sequence_checker_;
  bool is_screencast_ RTC_GUARDED_BY(sequence_checker_);
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_MEDIA_TEST_VIDEO_CAPTURER_VIDEO_TRACK_SOURCE_H_
