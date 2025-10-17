/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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
#include "pc/video_rtp_track_source.h"

#include <stddef.h>

#include <algorithm>

#include "rtc_base/checks.h"

namespace webrtc {

VideoRtpTrackSource::VideoRtpTrackSource(Callback* callback)
    : VideoTrackSource(true /* remote */), callback_(callback) {}

void VideoRtpTrackSource::ClearCallback() {
  RTC_DCHECK_RUN_ON(&worker_sequence_checker_);
  callback_ = nullptr;
}

rtc::VideoSourceInterface<VideoFrame>* VideoRtpTrackSource::source() {
  return &broadcaster_;
}
rtc::VideoSinkInterface<VideoFrame>* VideoRtpTrackSource::sink() {
  return &broadcaster_;
}

void VideoRtpTrackSource::BroadcastRecordableEncodedFrame(
    const RecordableEncodedFrame& frame) const {
  MutexLock lock(&mu_);
  for (rtc::VideoSinkInterface<RecordableEncodedFrame>* sink : encoded_sinks_) {
    sink->OnFrame(frame);
  }
}

bool VideoRtpTrackSource::SupportsEncodedOutput() const {
  return true;
}

void VideoRtpTrackSource::GenerateKeyFrame() {
  RTC_DCHECK_RUN_ON(&worker_sequence_checker_);
  if (callback_) {
    callback_->OnGenerateKeyFrame();
  }
}

void VideoRtpTrackSource::AddEncodedSink(
    rtc::VideoSinkInterface<RecordableEncodedFrame>* sink) {
  RTC_DCHECK_RUN_ON(&worker_sequence_checker_);
  RTC_DCHECK(sink);
  size_t size = 0;
  {
    MutexLock lock(&mu_);
    RTC_DCHECK(std::find(encoded_sinks_.begin(), encoded_sinks_.end(), sink) ==
               encoded_sinks_.end());
    encoded_sinks_.push_back(sink);
    size = encoded_sinks_.size();
  }
  if (size == 1 && callback_) {
    callback_->OnEncodedSinkEnabled(true);
  }
}

void VideoRtpTrackSource::RemoveEncodedSink(
    rtc::VideoSinkInterface<RecordableEncodedFrame>* sink) {
  RTC_DCHECK_RUN_ON(&worker_sequence_checker_);
  size_t size = 0;
  {
    MutexLock lock(&mu_);
    auto it = std::find(encoded_sinks_.begin(), encoded_sinks_.end(), sink);
    if (it != encoded_sinks_.end()) {
      encoded_sinks_.erase(it);
    }
    size = encoded_sinks_.size();
  }
  if (size == 0 && callback_) {
    callback_->OnEncodedSinkEnabled(false);
  }
}

}  // namespace webrtc
