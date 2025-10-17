/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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
#include "modules/video_coding/frame_helpers.h"

#include <utility>

#include "rtc_base/logging.h"

namespace webrtc {

namespace {
constexpr TimeDelta kMaxVideoDelay = TimeDelta::Millis(10000);
}

bool FrameHasBadRenderTiming(Timestamp render_time, Timestamp now) {
  // Zero render time means render immediately.
  if (render_time.IsZero()) {
    return false;
  }
  if (render_time < Timestamp::Zero()) {
    return true;
  }
  TimeDelta frame_delay = render_time - now;
  if (frame_delay.Abs() > kMaxVideoDelay) {
    RTC_LOG(LS_WARNING) << "Frame has bad render timing because it is out of "
                           "the delay bounds (frame_delay_ms="
                        << frame_delay.ms()
                        << ", kMaxVideoDelay_ms=" << kMaxVideoDelay.ms() << ")";
    return true;
  }
  return false;
}

bool TargetVideoDelayIsTooLarge(TimeDelta target_video_delay) {
  if (target_video_delay > kMaxVideoDelay) {
    RTC_LOG(LS_WARNING)
        << "Target video delay is too large. (target_video_delay_ms="
        << target_video_delay.ms()
        << ", kMaxVideoDelay_ms=" << kMaxVideoDelay.ms() << ")";
    return true;
  }
  return false;
}

std::unique_ptr<EncodedFrame> CombineAndDeleteFrames(
    absl::InlinedVector<std::unique_ptr<EncodedFrame>, 4> frames) {
  RTC_DCHECK(!frames.empty());

  if (frames.size() == 1) {
    return std::move(frames[0]);
  }

  size_t total_length = 0;
  for (const auto& frame : frames) {
    total_length += frame->size();
  }
  const EncodedFrame& last_frame = *frames.back();
  std::unique_ptr<EncodedFrame> first_frame = std::move(frames[0]);
  auto encoded_image_buffer = EncodedImageBuffer::Create(total_length);
  uint8_t* buffer = encoded_image_buffer->data();
  first_frame->SetSpatialLayerFrameSize(first_frame->SpatialIndex().value_or(0),
                                        first_frame->size());
  memcpy(buffer, first_frame->data(), first_frame->size());
  buffer += first_frame->size();

  // Spatial index of combined frame is set equal to spatial index of its top
  // spatial layer.
  first_frame->SetSpatialIndex(last_frame.SpatialIndex().value_or(0));

  first_frame->video_timing_mutable()->network2_timestamp_ms =
      last_frame.video_timing().network2_timestamp_ms;
  first_frame->video_timing_mutable()->receive_finish_ms =
      last_frame.video_timing().receive_finish_ms;

  // Append all remaining frames to the first one.
  for (size_t i = 1; i < frames.size(); ++i) {
    // Let |next_frame| fall out of scope so it is deleted after copying.
    std::unique_ptr<EncodedFrame> next_frame = std::move(frames[i]);
    first_frame->SetSpatialLayerFrameSize(
        next_frame->SpatialIndex().value_or(0), next_frame->size());
    memcpy(buffer, next_frame->data(), next_frame->size());
    buffer += next_frame->size();
  }
  first_frame->SetEncodedData(encoded_image_buffer);
  return first_frame;
}

}  // namespace webrtc
