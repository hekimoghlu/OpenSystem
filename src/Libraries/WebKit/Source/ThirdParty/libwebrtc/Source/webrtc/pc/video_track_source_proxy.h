/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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
#ifndef PC_VIDEO_TRACK_SOURCE_PROXY_H_
#define PC_VIDEO_TRACK_SOURCE_PROXY_H_

#include <optional>

#include "api/media_stream_interface.h"
#include "api/video/recordable_encoded_frame.h"
#include "api/video/video_frame.h"
#include "api/video/video_sink_interface.h"
#include "api/video/video_source_interface.h"
#include "api/video_track_source_constraints.h"
#include "pc/proxy.h"

namespace webrtc {

// Makes sure the real VideoTrackSourceInterface implementation is destroyed on
// the signaling thread and marshals all method calls to the signaling thread.
// TODO(deadbeef): Move this to .cc file. What threads methods are called on is
// an implementation detail.
BEGIN_PROXY_MAP(VideoTrackSource)

PROXY_PRIMARY_THREAD_DESTRUCTOR()
PROXY_CONSTMETHOD0(SourceState, state)
BYPASS_PROXY_CONSTMETHOD0(bool, remote)
BYPASS_PROXY_CONSTMETHOD0(bool, is_screencast)
PROXY_CONSTMETHOD0(std::optional<bool>, needs_denoising)
PROXY_METHOD1(bool, GetStats, Stats*)
PROXY_SECONDARY_METHOD2(void,
                        AddOrUpdateSink,
                        rtc::VideoSinkInterface<VideoFrame>*,
                        const rtc::VideoSinkWants&)
PROXY_SECONDARY_METHOD1(void, RemoveSink, rtc::VideoSinkInterface<VideoFrame>*)
PROXY_SECONDARY_METHOD0(void, RequestRefreshFrame)
PROXY_METHOD1(void, RegisterObserver, ObserverInterface*)
PROXY_METHOD1(void, UnregisterObserver, ObserverInterface*)
PROXY_CONSTMETHOD0(bool, SupportsEncodedOutput)
PROXY_SECONDARY_METHOD0(void, GenerateKeyFrame)
PROXY_SECONDARY_METHOD1(void,
                        AddEncodedSink,
                        rtc::VideoSinkInterface<RecordableEncodedFrame>*)
PROXY_SECONDARY_METHOD1(void,
                        RemoveEncodedSink,
                        rtc::VideoSinkInterface<RecordableEncodedFrame>*)
PROXY_SECONDARY_METHOD1(void,
                        ProcessConstraints,
                        const VideoTrackSourceConstraints&)
END_PROXY_MAP(VideoTrackSource)

}  // namespace webrtc

#endif  // PC_VIDEO_TRACK_SOURCE_PROXY_H_
