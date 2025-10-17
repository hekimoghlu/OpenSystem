/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
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
#ifndef PC_MEDIA_STREAM_PROXY_H_
#define PC_MEDIA_STREAM_PROXY_H_

#include <string>

#include "api/media_stream_interface.h"
#include "pc/proxy.h"

namespace webrtc {

// TODO(deadbeef): Move this to a .cc file. What threads methods are called on
// is an implementation detail.
BEGIN_PRIMARY_PROXY_MAP(MediaStream)
PROXY_PRIMARY_THREAD_DESTRUCTOR()
BYPASS_PROXY_CONSTMETHOD0(std::string, id)
PROXY_METHOD0(AudioTrackVector, GetAudioTracks)
PROXY_METHOD0(VideoTrackVector, GetVideoTracks)
PROXY_METHOD1(rtc::scoped_refptr<AudioTrackInterface>,
              FindAudioTrack,
              const std::string&)
PROXY_METHOD1(rtc::scoped_refptr<VideoTrackInterface>,
              FindVideoTrack,
              const std::string&)
PROXY_METHOD1(bool, AddTrack, rtc::scoped_refptr<AudioTrackInterface>)
PROXY_METHOD1(bool, AddTrack, rtc::scoped_refptr<VideoTrackInterface>)
PROXY_METHOD1(bool, RemoveTrack, rtc::scoped_refptr<AudioTrackInterface>)
PROXY_METHOD1(bool, RemoveTrack, rtc::scoped_refptr<VideoTrackInterface>)
PROXY_METHOD1(void, RegisterObserver, ObserverInterface*)
PROXY_METHOD1(void, UnregisterObserver, ObserverInterface*)
END_PROXY_MAP(MediaStream)

}  // namespace webrtc

#endif  // PC_MEDIA_STREAM_PROXY_H_
