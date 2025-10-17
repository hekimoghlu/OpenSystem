/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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
#include "test/mac_capturer.h"

#import "sdk/objc/base/RTCVideoCapturer.h"
#import "sdk/objc/components/capturer/RTCCameraVideoCapturer.h"
#import "sdk/objc/native/api/video_capturer.h"
#import "sdk/objc/native/src/objc_frame_buffer.h"

@interface RTCTestVideoSourceAdapter : NSObject <RTC_OBJC_TYPE (RTCVideoCapturerDelegate)>
@property(nonatomic) webrtc::test::MacCapturer *capturer;
@end

@implementation RTCTestVideoSourceAdapter
@synthesize capturer = _capturer;

- (void)capturer:(RTC_OBJC_TYPE(RTCVideoCapturer) *)capturer
    didCaptureVideoFrame:(RTC_OBJC_TYPE(RTCVideoFrame) *)frame {
  const int64_t timestamp_us = frame.timeStampNs / rtc::kNumNanosecsPerMicrosec;
  rtc::scoped_refptr<webrtc::VideoFrameBuffer> buffer =
      rtc::make_ref_counted<webrtc::ObjCFrameBuffer>(frame.buffer);
  _capturer->OnFrame(webrtc::VideoFrame::Builder()
                         .set_video_frame_buffer(buffer)
                         .set_rotation(webrtc::kVideoRotation_0)
                         .set_timestamp_us(timestamp_us)
                         .build());
}

@end

namespace {

AVCaptureDeviceFormat *SelectClosestFormat(AVCaptureDevice *device, size_t width, size_t height) {
  NSArray<AVCaptureDeviceFormat *> *formats =
      [RTC_OBJC_TYPE(RTCCameraVideoCapturer) supportedFormatsForDevice:device];
  AVCaptureDeviceFormat *selectedFormat = nil;
  int currentDiff = INT_MAX;
  for (AVCaptureDeviceFormat *format in formats) {
    CMVideoDimensions dimension = CMVideoFormatDescriptionGetDimensions(format.formatDescription);
    int diff =
        std::abs((int64_t)width - dimension.width) + std::abs((int64_t)height - dimension.height);
    if (diff < currentDiff) {
      selectedFormat = format;
      currentDiff = diff;
    }
  }
  return selectedFormat;
}

}  // namespace

namespace webrtc {
namespace test {

MacCapturer::MacCapturer(size_t width,
                         size_t height,
                         size_t target_fps,
                         size_t capture_device_index) {
  width_ = width;
  height_ = height;
  RTCTestVideoSourceAdapter *adapter = [[RTCTestVideoSourceAdapter alloc] init];
  adapter_ = (__bridge_retained void *)adapter;
  adapter.capturer = this;

  RTC_OBJC_TYPE(RTCCameraVideoCapturer) *capturer =
      [[RTC_OBJC_TYPE(RTCCameraVideoCapturer) alloc] initWithDelegate:adapter];
  capturer_ = (__bridge_retained void *)capturer;

  AVCaptureDevice *device =
      [[RTC_OBJC_TYPE(RTCCameraVideoCapturer) captureDevices] objectAtIndex:capture_device_index];
  AVCaptureDeviceFormat *format = SelectClosestFormat(device, width, height);
  [capturer startCaptureWithDevice:device format:format fps:target_fps];
}

MacCapturer *MacCapturer::Create(size_t width,
                                 size_t height,
                                 size_t target_fps,
                                 size_t capture_device_index) {
  return new MacCapturer(width, height, target_fps, capture_device_index);
}

void MacCapturer::Destroy() {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
  RTCTestVideoSourceAdapter *adapter = (__bridge_transfer RTCTestVideoSourceAdapter *)adapter_;
  RTC_OBJC_TYPE(RTCCameraVideoCapturer) *capturer =
      (__bridge_transfer RTC_OBJC_TYPE(RTCCameraVideoCapturer) *)capturer_;
  [capturer stopCapture];
#pragma clang diagnostic pop
}

MacCapturer::~MacCapturer() {
  Destroy();
}

void MacCapturer::OnFrame(const VideoFrame &frame) {
  TestVideoCapturer::OnFrame(frame);
}

}  // namespace test
}  // namespace webrtc
