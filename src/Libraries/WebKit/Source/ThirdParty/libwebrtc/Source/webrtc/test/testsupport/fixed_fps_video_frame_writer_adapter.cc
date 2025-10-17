/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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
#include "test/testsupport/fixed_fps_video_frame_writer_adapter.h"

#include <cmath>
#include <optional>
#include <utility>

#include "api/units/time_delta.h"
#include "api/video/video_sink_interface.h"
#include "rtc_base/checks.h"
#include "test/testsupport/video_frame_writer.h"

namespace webrtc {
namespace test {
namespace {

constexpr TimeDelta kOneSecond = TimeDelta::Seconds(1);

}  // namespace

FixedFpsVideoFrameWriterAdapter::FixedFpsVideoFrameWriterAdapter(
    int fps,
    Clock* clock,
    std::unique_ptr<VideoFrameWriter> delegate)
    : inter_frame_interval_(kOneSecond / fps),
      clock_(clock),
      delegate_(std::move(delegate)) {}

FixedFpsVideoFrameWriterAdapter::~FixedFpsVideoFrameWriterAdapter() {
  Close();
}

void FixedFpsVideoFrameWriterAdapter::Close() {
  if (is_closed_) {
    return;
  }
  is_closed_ = true;
  if (!last_frame_.has_value()) {
    return;
  }
  Timestamp now = Now();
  RTC_CHECK(WriteMissedSlotsExceptLast(now));
  RTC_CHECK(delegate_->WriteFrame(*last_frame_));
  delegate_->Close();
}

bool FixedFpsVideoFrameWriterAdapter::WriteFrame(const VideoFrame& frame) {
  RTC_CHECK(!is_closed_);
  Timestamp now = Now();
  if (!last_frame_.has_value()) {
    RTC_CHECK(!last_frame_time_.IsFinite());
    last_frame_ = frame;
    last_frame_time_ = now;
    return true;
  }

  RTC_CHECK(last_frame_time_.IsFinite());

  if (last_frame_time_ > now) {
    // New frame was recevied before expected time "slot" for current
    // `last_frame_` came => just replace current `last_frame_` with
    // received `frame`.
    RTC_CHECK_LE(last_frame_time_ - now, inter_frame_interval_ / 2);
    last_frame_ = frame;
    return true;
  }

  if (!WriteMissedSlotsExceptLast(now)) {
    return false;
  }

  if (now - last_frame_time_ < inter_frame_interval_ / 2) {
    // New frame was received closer to the expected time "slot" for current
    // `last_frame_` than to the next "slot" => just replace current
    // `last_frame_` with received `frame`.
    last_frame_ = frame;
    return true;
  }

  if (!delegate_->WriteFrame(*last_frame_)) {
    return false;
  }
  last_frame_ = frame;
  last_frame_time_ = last_frame_time_ + inter_frame_interval_;
  return true;
}

bool FixedFpsVideoFrameWriterAdapter::WriteMissedSlotsExceptLast(
    Timestamp now) {
  RTC_CHECK(last_frame_time_.IsFinite());
  while (now - last_frame_time_ > inter_frame_interval_) {
    if (!delegate_->WriteFrame(*last_frame_)) {
      return false;
    }
    last_frame_time_ = last_frame_time_ + inter_frame_interval_;
  }
  return true;
}

Timestamp FixedFpsVideoFrameWriterAdapter::Now() const {
  return clock_->CurrentTime();
}

}  // namespace test
}  // namespace webrtc
