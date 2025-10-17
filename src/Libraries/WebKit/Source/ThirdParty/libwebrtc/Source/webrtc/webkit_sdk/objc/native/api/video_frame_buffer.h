/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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
#ifndef SDK_OBJC_NATIVE_API_VIDEO_FRAME_BUFFER_H_
#define SDK_OBJC_NATIVE_API_VIDEO_FRAME_BUFFER_H_

#import "base/RTCVideoFrameBuffer.h"

#include "api/scoped_refptr.h"
#include "common_video/include/video_frame_buffer.h"

namespace webrtc {

id<RTCVideoFrameBuffer> NativeToObjCVideoFrameBuffer(
    const rtc::scoped_refptr<VideoFrameBuffer>& buffer);

}  // namespace webrtc

#endif  // SDK_OBJC_NATIVE_API_VIDEO_FRAME_BUFFER_H_
