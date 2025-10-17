/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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
#ifndef SDK_OBJC_NATIVE_SRC_OBJC_VIDEO_RENDERER_H_
#define SDK_OBJC_NATIVE_SRC_OBJC_VIDEO_RENDERER_H_

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>

#include "api/video/video_frame.h"
#include "api/video/video_sink_interface.h"

@protocol RTCVideoRenderer;

namespace webrtc {

class ObjCVideoRenderer : public rtc::VideoSinkInterface<VideoFrame> {
 public:
  ObjCVideoRenderer(id<RTCVideoRenderer> renderer);
  void OnFrame(const VideoFrame& nativeVideoFrame) override;

 private:
  id<RTCVideoRenderer> renderer_;
  CGSize size_;
};

}  // namespace webrtc

#endif  // SDK_OBJC_NATIVE_SRC_OBJC_VIDEO_RENDERER_H_
