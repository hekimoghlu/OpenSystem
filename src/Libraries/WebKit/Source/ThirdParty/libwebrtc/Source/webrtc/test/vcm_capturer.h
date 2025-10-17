/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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
#ifndef TEST_VCM_CAPTURER_H_
#define TEST_VCM_CAPTURER_H_

#include <memory>
#include <vector>

#include "api/scoped_refptr.h"
#include "modules/video_capture/video_capture.h"
#include "rtc_base/logging.h"
#include "test/test_video_capturer.h"

namespace webrtc {
namespace test {

class VcmCapturer : public TestVideoCapturer,
                    public rtc::VideoSinkInterface<VideoFrame> {
 public:
  static VcmCapturer* Create(size_t width,
                             size_t height,
                             size_t target_fps,
                             size_t capture_device_index);
  virtual ~VcmCapturer();

  void Start() override {
    RTC_LOG(LS_WARNING) << "Capturer doesn't support resume/pause and always "
                           "produces the video";
  }
  void Stop() override {
    RTC_LOG(LS_WARNING) << "Capturer doesn't support resume/pause and always "
                           "produces the video";
  }

  void OnFrame(const VideoFrame& frame) override;

  int GetFrameWidth() const override { return static_cast<int>(width_); }
  int GetFrameHeight() const override { return static_cast<int>(height_); }

 private:
  VcmCapturer();
  bool Init(size_t width,
            size_t height,
            size_t target_fps,
            size_t capture_device_index);
  void Destroy();

  size_t width_;
  size_t height_;
  rtc::scoped_refptr<VideoCaptureModule> vcm_;
  VideoCaptureCapability capability_;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_VCM_CAPTURER_H_
