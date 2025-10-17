/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
#include "media/base/fake_frame_source.h"

#include "api/scoped_refptr.h"
#include "api/video/i420_buffer.h"
#include "api/video/video_frame_buffer.h"
#include "rtc_base/checks.h"
#include "rtc_base/time_utils.h"

namespace cricket {

FakeFrameSource::FakeFrameSource(int width,
                                 int height,
                                 int interval_us,
                                 int64_t timestamp_offset_us)
    : width_(width),
      height_(height),
      interval_us_(interval_us),
      next_timestamp_us_(timestamp_offset_us) {
  RTC_CHECK_GT(width_, 0);
  RTC_CHECK_GT(height_, 0);
  RTC_CHECK_GT(interval_us_, 0);
  RTC_CHECK_GE(next_timestamp_us_, 0);
}

FakeFrameSource::FakeFrameSource(int width, int height, int interval_us)
    : FakeFrameSource(width, height, interval_us, rtc::TimeMicros()) {}

webrtc::VideoRotation FakeFrameSource::GetRotation() const {
  return rotation_;
}

void FakeFrameSource::SetRotation(webrtc::VideoRotation rotation) {
  rotation_ = rotation;
}

webrtc::VideoFrame FakeFrameSource::GetFrameRotationApplied() {
  switch (rotation_) {
    case webrtc::kVideoRotation_0:
    case webrtc::kVideoRotation_180:
      return GetFrame(width_, height_, webrtc::kVideoRotation_0, interval_us_);
    case webrtc::kVideoRotation_90:
    case webrtc::kVideoRotation_270:
      return GetFrame(height_, width_, webrtc::kVideoRotation_0, interval_us_);
  }
  RTC_DCHECK_NOTREACHED() << "Invalid rotation value: "
                          << static_cast<int>(rotation_);
  // Without this return, the Windows Visual Studio compiler complains
  // "not all control paths return a value".
  return GetFrame();
}

webrtc::VideoFrame FakeFrameSource::GetFrame() {
  return GetFrame(width_, height_, rotation_, interval_us_);
}

webrtc::VideoFrame FakeFrameSource::GetFrame(int width,
                                             int height,
                                             webrtc::VideoRotation rotation,
                                             int interval_us) {
  RTC_CHECK_GT(width, 0);
  RTC_CHECK_GT(height, 0);
  RTC_CHECK_GT(interval_us, 0);

  rtc::scoped_refptr<webrtc::I420Buffer> buffer(
      webrtc::I420Buffer::Create(width, height));

  buffer->InitializeData();
  webrtc::VideoFrame frame = webrtc::VideoFrame::Builder()
                                 .set_video_frame_buffer(buffer)
                                 .set_rotation(rotation)
                                 .set_timestamp_us(next_timestamp_us_)
                                 .build();

  next_timestamp_us_ += interval_us;
  return frame;
}

}  // namespace cricket
