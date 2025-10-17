/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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
#include "api/test/video/test_video_track_source.h"

#include <optional>
#include <string>
#include <utility>

#include "api/media_stream_interface.h"
#include "api/sequence_checker.h"
#include "api/video/video_frame.h"
#include "api/video/video_sink_interface.h"
#include "api/video/video_source_interface.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

TestVideoTrackSource::TestVideoTrackSource(
    bool remote,
    std::optional<std::string> stream_label)
    : stream_label_(std::move(stream_label)),
      state_(kInitializing),
      remote_(remote) {
  worker_thread_checker_.Detach();
  signaling_thread_checker_.Detach();
}

VideoTrackSourceInterface::SourceState TestVideoTrackSource::state() const {
  RTC_DCHECK_RUN_ON(&signaling_thread_checker_);
  return state_;
}

void TestVideoTrackSource::SetState(SourceState new_state) {
  RTC_DCHECK_RUN_ON(&signaling_thread_checker_);
  if (state_ != new_state) {
    state_ = new_state;
    FireOnChanged();
  }
}

void TestVideoTrackSource::AddOrUpdateSink(
    rtc::VideoSinkInterface<VideoFrame>* sink,
    const rtc::VideoSinkWants& wants) {
  RTC_DCHECK(worker_thread_checker_.IsCurrent());
  source()->AddOrUpdateSink(sink, wants);
}

void TestVideoTrackSource::RemoveSink(
    rtc::VideoSinkInterface<VideoFrame>* sink) {
  RTC_DCHECK(worker_thread_checker_.IsCurrent());
  source()->RemoveSink(sink);
}

}  // namespace test
}  // namespace webrtc
