/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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
// This file includes proxy classes for tracks. The purpose is
// to make sure tracks are only accessed from the signaling thread.

#ifndef PC_MEDIA_STREAM_TRACK_PROXY_H_
#define PC_MEDIA_STREAM_TRACK_PROXY_H_

#include <string>

#include "api/media_stream_interface.h"
#include "pc/proxy.h"

namespace webrtc {

// TODO(deadbeef): Move this to .cc file. What threads methods are called on is
// an implementation detail.
BEGIN_PRIMARY_PROXY_MAP(AudioTrack)
PROXY_PRIMARY_THREAD_DESTRUCTOR()
BYPASS_PROXY_CONSTMETHOD0(std::string, kind)
BYPASS_PROXY_CONSTMETHOD0(std::string, id)
PROXY_CONSTMETHOD0(TrackState, state)
PROXY_CONSTMETHOD0(bool, enabled)
BYPASS_PROXY_CONSTMETHOD0(AudioSourceInterface*, GetSource)
PROXY_METHOD1(void, AddSink, AudioTrackSinkInterface*)
PROXY_METHOD1(void, RemoveSink, AudioTrackSinkInterface*)
PROXY_METHOD1(bool, GetSignalLevel, int*)
PROXY_METHOD0(rtc::scoped_refptr<AudioProcessorInterface>, GetAudioProcessor)
PROXY_METHOD1(bool, set_enabled, bool)
PROXY_METHOD1(void, RegisterObserver, ObserverInterface*)
PROXY_METHOD1(void, UnregisterObserver, ObserverInterface*)
END_PROXY_MAP(AudioTrack)

BEGIN_PROXY_MAP(VideoTrack)
PROXY_PRIMARY_THREAD_DESTRUCTOR()
BYPASS_PROXY_CONSTMETHOD0(std::string, kind)
BYPASS_PROXY_CONSTMETHOD0(std::string, id)
PROXY_SECONDARY_CONSTMETHOD0(TrackState, state)
PROXY_CONSTMETHOD0(bool, enabled)
PROXY_METHOD1(bool, set_enabled, bool)
PROXY_CONSTMETHOD0(ContentHint, content_hint)
PROXY_METHOD1(void, set_content_hint, ContentHint)
PROXY_SECONDARY_METHOD2(void,
                        AddOrUpdateSink,
                        rtc::VideoSinkInterface<VideoFrame>*,
                        const rtc::VideoSinkWants&)
PROXY_SECONDARY_METHOD1(void, RemoveSink, rtc::VideoSinkInterface<VideoFrame>*)
PROXY_SECONDARY_METHOD0(void, RequestRefreshFrame)
BYPASS_PROXY_CONSTMETHOD0(VideoTrackSourceInterface*, GetSource)

PROXY_METHOD1(void, RegisterObserver, ObserverInterface*)
PROXY_METHOD1(void, UnregisterObserver, ObserverInterface*)
END_PROXY_MAP(VideoTrack)

}  // namespace webrtc

#endif  // PC_MEDIA_STREAM_TRACK_PROXY_H_
