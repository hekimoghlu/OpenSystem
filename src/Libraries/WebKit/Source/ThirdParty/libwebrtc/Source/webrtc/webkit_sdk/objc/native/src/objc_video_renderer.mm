/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 29, 2024.
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
#include "sdk/objc/native/src/objc_video_renderer.h"

#import "base/RTCVideoFrame.h"
#import "base/RTCVideoRenderer.h"

#include "sdk/objc/native/src/objc_video_frame.h"

namespace webrtc {

ObjCVideoRenderer::ObjCVideoRenderer(id<RTCVideoRenderer> renderer)
    : renderer_(renderer), size_(CGSizeZero) {}

void ObjCVideoRenderer::OnFrame(const VideoFrame& nativeVideoFrame) {
  RTCVideoFrame* videoFrame = ToObjCVideoFrame(nativeVideoFrame);

  CGSize current_size = (videoFrame.rotation % 180 == 0) ?
      CGSizeMake(videoFrame.width, videoFrame.height) :
      CGSizeMake(videoFrame.height, videoFrame.width);

  if (!CGSizeEqualToSize(size_, current_size)) {
    size_ = current_size;
    [renderer_ setSize:size_];
  }
  [renderer_ renderFrame:videoFrame];
}

}  // namespace webrtc
