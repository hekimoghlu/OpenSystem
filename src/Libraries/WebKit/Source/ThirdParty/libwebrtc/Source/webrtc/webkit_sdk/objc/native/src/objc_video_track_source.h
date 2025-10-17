/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
#ifndef SDK_OBJC_CLASSES_VIDEO_OBJC_VIDEO_TRACK_SOURCE_H_
#define SDK_OBJC_CLASSES_VIDEO_OBJC_VIDEO_TRACK_SOURCE_H_

#import "base/RTCVideoCapturer.h"

#include "base/RTCMacros.h"
#include "media/base/adapted_video_track_source.h"
#include "rtc_base/timestamp_aligner.h"

RTC_FWD_DECL_OBJC_CLASS(RTCVideoFrame);

@interface RTCObjCVideoSourceAdapter : NSObject<RTCVideoCapturerDelegate>
@end

namespace webrtc {

class ObjCVideoTrackSource : public rtc::AdaptedVideoTrackSource {
 public:
  ObjCVideoTrackSource();
  explicit ObjCVideoTrackSource(RTCObjCVideoSourceAdapter* adapter);

  // This class can not be used for implementing screen casting. Hopefully, this
  // function will be removed before we add that to iOS/Mac.
  bool is_screencast() const override;

  // Indicates that the encoder should denoise video before encoding it.
  // If it is not set, the default configuration is used which is different
  // depending on video codec.
  std::optional<bool> needs_denoising() const override;

  SourceState state() const override;

  bool remote() const override;

  void OnCapturedFrame(RTCVideoFrame* frame);

  // Called by RTCVideoSource.
  void OnOutputFormatRequest(int width, int height, int fps);

 private:
  rtc::VideoBroadcaster broadcaster_;
  rtc::TimestampAligner timestamp_aligner_;

  RTCObjCVideoSourceAdapter* adapter_;
};

}  // namespace webrtc

#endif  // SDK_OBJC_CLASSES_VIDEO_OBJC_VIDEO_TRACK_SOURCE_H_
