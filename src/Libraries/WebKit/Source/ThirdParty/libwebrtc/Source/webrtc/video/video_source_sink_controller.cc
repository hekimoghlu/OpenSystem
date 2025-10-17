/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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
#include "video/video_source_sink_controller.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "rtc_base/logging.h"
#include "rtc_base/numerics/safe_conversions.h"
#include "rtc_base/strings/string_builder.h"

namespace webrtc {

VideoSourceSinkController::VideoSourceSinkController(
    rtc::VideoSinkInterface<VideoFrame>* sink,
    rtc::VideoSourceInterface<VideoFrame>* source)
    : sink_(sink), source_(source) {
  RTC_DCHECK(sink_);
}

VideoSourceSinkController::~VideoSourceSinkController() {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
}

void VideoSourceSinkController::SetSource(
    rtc::VideoSourceInterface<VideoFrame>* source) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);

  rtc::VideoSourceInterface<VideoFrame>* old_source = source_;
  source_ = source;

  if (old_source != source && old_source)
    old_source->RemoveSink(sink_);

  if (!source)
    return;

  source->AddOrUpdateSink(sink_, CurrentSettingsToSinkWants());
}

bool VideoSourceSinkController::HasSource() const {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  return source_ != nullptr;
}

void VideoSourceSinkController::RequestRefreshFrame() {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  if (source_)
    source_->RequestRefreshFrame();
}

void VideoSourceSinkController::PushSourceSinkSettings() {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  if (!source_)
    return;
  rtc::VideoSinkWants wants = CurrentSettingsToSinkWants();
  source_->AddOrUpdateSink(sink_, wants);
}

VideoSourceRestrictions VideoSourceSinkController::restrictions() const {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  return restrictions_;
}

std::optional<size_t> VideoSourceSinkController::pixels_per_frame_upper_limit()
    const {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  return pixels_per_frame_upper_limit_;
}

std::optional<double> VideoSourceSinkController::frame_rate_upper_limit()
    const {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  return frame_rate_upper_limit_;
}

bool VideoSourceSinkController::rotation_applied() const {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  return rotation_applied_;
}

int VideoSourceSinkController::resolution_alignment() const {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  return resolution_alignment_;
}

const std::vector<rtc::VideoSinkWants::FrameSize>&
VideoSourceSinkController::resolutions() const {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  return resolutions_;
}

bool VideoSourceSinkController::active() const {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  return active_;
}

std::optional<rtc::VideoSinkWants::FrameSize>
VideoSourceSinkController::scale_resolution_down_to() const {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  return scale_resolution_down_to_;
}

void VideoSourceSinkController::SetRestrictions(
    VideoSourceRestrictions restrictions) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  restrictions_ = std::move(restrictions);
}

void VideoSourceSinkController::SetPixelsPerFrameUpperLimit(
    std::optional<size_t> pixels_per_frame_upper_limit) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  pixels_per_frame_upper_limit_ = std::move(pixels_per_frame_upper_limit);
}

void VideoSourceSinkController::SetFrameRateUpperLimit(
    std::optional<double> frame_rate_upper_limit) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  frame_rate_upper_limit_ = std::move(frame_rate_upper_limit);
}

void VideoSourceSinkController::SetRotationApplied(bool rotation_applied) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  rotation_applied_ = rotation_applied;
}

void VideoSourceSinkController::SetResolutionAlignment(
    int resolution_alignment) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  resolution_alignment_ = resolution_alignment;
}

void VideoSourceSinkController::SetResolutions(
    std::vector<rtc::VideoSinkWants::FrameSize> resolutions) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  resolutions_ = std::move(resolutions);
}

void VideoSourceSinkController::SetActive(bool active) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  active_ = active;
}

void VideoSourceSinkController::SetScaleResolutionDownTo(
    std::optional<rtc::VideoSinkWants::FrameSize> scale_resolution_down_to) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  scale_resolution_down_to_ = std::move(scale_resolution_down_to);
}

// RTC_EXCLUSIVE_LOCKS_REQUIRED(sequence_checker_)
rtc::VideoSinkWants VideoSourceSinkController::CurrentSettingsToSinkWants()
    const {
  rtc::VideoSinkWants wants;
  wants.rotation_applied = rotation_applied_;
  // `wants.black_frames` is not used, it always has its default value false.
  wants.max_pixel_count =
      rtc::dchecked_cast<int>(restrictions_.max_pixels_per_frame().value_or(
          std::numeric_limits<int>::max()));
  wants.target_pixel_count =
      restrictions_.target_pixels_per_frame().has_value()
          ? std::optional<int>(rtc::dchecked_cast<int>(
                restrictions_.target_pixels_per_frame().value()))
          : std::nullopt;
  wants.max_framerate_fps =
      restrictions_.max_frame_rate().has_value()
          ? static_cast<int>(restrictions_.max_frame_rate().value())
          : std::numeric_limits<int>::max();
  wants.resolution_alignment = resolution_alignment_;
  wants.max_pixel_count =
      std::min(wants.max_pixel_count,
               rtc::dchecked_cast<int>(pixels_per_frame_upper_limit_.value_or(
                   std::numeric_limits<int>::max())));
  wants.max_framerate_fps =
      std::min(wants.max_framerate_fps,
               frame_rate_upper_limit_.has_value()
                   ? static_cast<int>(frame_rate_upper_limit_.value())
                   : std::numeric_limits<int>::max());
  wants.resolutions = resolutions_;
  wants.is_active = active_;
  wants.requested_resolution = scale_resolution_down_to_;
  return wants;
}

}  // namespace webrtc
