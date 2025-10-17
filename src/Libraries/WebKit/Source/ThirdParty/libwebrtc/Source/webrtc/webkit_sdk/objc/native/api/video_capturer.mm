/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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
#include "sdk/objc/native/api/video_capturer.h"

#include "absl/memory/memory.h"
#include "api/video_track_source_proxy.h"
#include "sdk/objc/native/src/objc_video_track_source.h"

namespace webrtc {

rtc::scoped_refptr<webrtc::VideoTrackSourceInterface> ObjCToNativeVideoCapturer(
    RTCVideoCapturer *objc_video_capturer,
    rtc::Thread *signaling_thread,
    rtc::Thread *worker_thread) {
  RTCObjCVideoSourceAdapter *adapter = [[RTCObjCVideoSourceAdapter alloc] init];
  rtc::scoped_refptr<webrtc::ObjCVideoTrackSource> objc_video_track_source(
      new rtc::RefCountedObject<webrtc::ObjCVideoTrackSource>(adapter));
  rtc::scoped_refptr<webrtc::VideoTrackSourceInterface> video_source =
      webrtc::VideoTrackSourceProxy::Create(
          signaling_thread, worker_thread, objc_video_track_source);

  objc_video_capturer.delegate = adapter;

  return video_source;
}

}  // namespace webrtc
