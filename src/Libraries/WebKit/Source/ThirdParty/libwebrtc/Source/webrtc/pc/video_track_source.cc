/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
#include "pc/video_track_source.h"

#include "rtc_base/checks.h"

namespace webrtc {

VideoTrackSource::VideoTrackSource(bool remote)
    : state_(kInitializing), remote_(remote) {}

void VideoTrackSource::SetState(SourceState new_state) {
  RTC_DCHECK_RUN_ON(&signaling_thread_checker_);
  if (state_ != new_state) {
    state_ = new_state;
    FireOnChanged();
  }
}

void VideoTrackSource::AddOrUpdateSink(
    rtc::VideoSinkInterface<VideoFrame>* sink,
    const rtc::VideoSinkWants& wants) {
  RTC_DCHECK(worker_thread_checker_.IsCurrent());
  source()->AddOrUpdateSink(sink, wants);
}

void VideoTrackSource::RemoveSink(rtc::VideoSinkInterface<VideoFrame>* sink) {
  RTC_DCHECK(worker_thread_checker_.IsCurrent());
  source()->RemoveSink(sink);
}

}  //  namespace webrtc
