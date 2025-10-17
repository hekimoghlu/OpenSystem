/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 10, 2024.
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
#ifndef MODULES_VIDEO_CAPTURE_VIDEO_CAPTURE_DEFINES_H_
#define MODULES_VIDEO_CAPTURE_VIDEO_CAPTURE_DEFINES_H_

#include "api/video/video_frame.h"
#include "common_video/libyuv/include/webrtc_libyuv.h"

namespace webrtc {

enum {
  kVideoCaptureUniqueNameLength = 1024
};  // Max unique capture device name lenght
enum { kVideoCaptureDeviceNameLength = 256 };  // Max capture device name lenght
enum { kVideoCaptureProductIdLength = 128 };   // Max product id length

struct VideoCaptureCapability {
  int32_t width;
  int32_t height;
  int32_t maxFPS;
  VideoType videoType;
  bool interlaced;

  VideoCaptureCapability() {
    width = 0;
    height = 0;
    maxFPS = 0;
    videoType = VideoType::kUnknown;
    interlaced = false;
  }
  bool operator!=(const VideoCaptureCapability& other) const {
    if (width != other.width)
      return true;
    if (height != other.height)
      return true;
    if (maxFPS != other.maxFPS)
      return true;
    if (videoType != other.videoType)
      return true;
    if (interlaced != other.interlaced)
      return true;
    return false;
  }
  bool operator==(const VideoCaptureCapability& other) const {
    return !operator!=(other);
  }
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CAPTURE_VIDEO_CAPTURE_DEFINES_H_
