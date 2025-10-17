/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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
#import "RTCVideoRendererAdapter+Private.h"
#import "base/RTCVideoFrame.h"

#include <memory>

#include "sdk/objc/native/api/video_frame.h"

namespace webrtc {

class VideoRendererAdapter
    : public rtc::VideoSinkInterface<webrtc::VideoFrame> {
 public:
  VideoRendererAdapter(RTCVideoRendererAdapter* adapter) {
    adapter_ = adapter;
    size_ = CGSizeZero;
  }

  void OnFrame(const webrtc::VideoFrame& nativeVideoFrame) override {
    RTCVideoFrame* videoFrame = NativeToObjCVideoFrame(nativeVideoFrame);

    CGSize current_size = (videoFrame.rotation % 180 == 0)
                              ? CGSizeMake(videoFrame.width, videoFrame.height)
                              : CGSizeMake(videoFrame.height, videoFrame.width);

    if (!CGSizeEqualToSize(size_, current_size)) {
      size_ = current_size;
      [adapter_.videoRenderer setSize:size_];
    }
    [adapter_.videoRenderer renderFrame:videoFrame];
  }

 private:
  __weak RTCVideoRendererAdapter *adapter_;
  CGSize size_;
};
}

@implementation RTCVideoRendererAdapter {
  std::unique_ptr<webrtc::VideoRendererAdapter> _adapter;
}

@synthesize videoRenderer = _videoRenderer;

- (instancetype)initWithNativeRenderer:(id<RTCVideoRenderer>)videoRenderer {
  NSParameterAssert(videoRenderer);
  if (self = [super init]) {
    _videoRenderer = videoRenderer;
    _adapter.reset(new webrtc::VideoRendererAdapter(self));
  }
  return self;
}

- (rtc::VideoSinkInterface<webrtc::VideoFrame> *)nativeVideoRenderer {
  return _adapter.get();
}

@end
