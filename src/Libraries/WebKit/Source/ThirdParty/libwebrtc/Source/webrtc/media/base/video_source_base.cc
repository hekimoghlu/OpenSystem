/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 19, 2025.
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
#include "media/base/video_source_base.h"

#include <algorithm>

#include "absl/algorithm/container.h"
#include "rtc_base/checks.h"

namespace rtc {

VideoSourceBase::VideoSourceBase() = default;
VideoSourceBase::~VideoSourceBase() = default;

void VideoSourceBase::AddOrUpdateSink(
    VideoSinkInterface<webrtc::VideoFrame>* sink,
    const VideoSinkWants& wants) {
  RTC_DCHECK(sink != nullptr);

  SinkPair* sink_pair = FindSinkPair(sink);
  if (!sink_pair) {
    sinks_.push_back(SinkPair(sink, wants));
  } else {
    sink_pair->wants = wants;
  }
}

void VideoSourceBase::RemoveSink(VideoSinkInterface<webrtc::VideoFrame>* sink) {
  RTC_DCHECK(sink != nullptr);
  RTC_DCHECK(FindSinkPair(sink));
  sinks_.erase(std::remove_if(sinks_.begin(), sinks_.end(),
                              [sink](const SinkPair& sink_pair) {
                                return sink_pair.sink == sink;
                              }),
               sinks_.end());
}

VideoSourceBase::SinkPair* VideoSourceBase::FindSinkPair(
    const VideoSinkInterface<webrtc::VideoFrame>* sink) {
  auto sink_pair_it = absl::c_find_if(
      sinks_,
      [sink](const SinkPair& sink_pair) { return sink_pair.sink == sink; });
  if (sink_pair_it != sinks_.end()) {
    return &*sink_pair_it;
  }
  return nullptr;
}

VideoSourceBaseGuarded::VideoSourceBaseGuarded() = default;
VideoSourceBaseGuarded::~VideoSourceBaseGuarded() = default;

void VideoSourceBaseGuarded::AddOrUpdateSink(
    VideoSinkInterface<webrtc::VideoFrame>* sink,
    const VideoSinkWants& wants) {
  RTC_DCHECK_RUN_ON(&source_sequence_);
  RTC_DCHECK(sink != nullptr);

  SinkPair* sink_pair = FindSinkPair(sink);
  if (!sink_pair) {
    sinks_.push_back(SinkPair(sink, wants));
  } else {
    sink_pair->wants = wants;
  }
}

void VideoSourceBaseGuarded::RemoveSink(
    VideoSinkInterface<webrtc::VideoFrame>* sink) {
  RTC_DCHECK_RUN_ON(&source_sequence_);
  RTC_DCHECK(sink != nullptr);
  RTC_DCHECK(FindSinkPair(sink));
  sinks_.erase(std::remove_if(sinks_.begin(), sinks_.end(),
                              [sink](const SinkPair& sink_pair) {
                                return sink_pair.sink == sink;
                              }),
               sinks_.end());
}

VideoSourceBaseGuarded::SinkPair* VideoSourceBaseGuarded::FindSinkPair(
    const VideoSinkInterface<webrtc::VideoFrame>* sink) {
  RTC_DCHECK_RUN_ON(&source_sequence_);
  auto sink_pair_it = absl::c_find_if(
      sinks_,
      [sink](const SinkPair& sink_pair) { return sink_pair.sink == sink; });
  if (sink_pair_it != sinks_.end()) {
    return &*sink_pair_it;
  }
  return nullptr;
}

const std::vector<VideoSourceBaseGuarded::SinkPair>&
VideoSourceBaseGuarded::sink_pairs() const {
  RTC_DCHECK_RUN_ON(&source_sequence_);
  return sinks_;
}

}  // namespace rtc
