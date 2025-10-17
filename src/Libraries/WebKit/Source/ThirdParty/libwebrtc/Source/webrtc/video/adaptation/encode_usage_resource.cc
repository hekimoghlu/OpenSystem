/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 12, 2023.
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
#include "video/adaptation/encode_usage_resource.h"

#include <limits>
#include <utility>

#include "rtc_base/checks.h"

namespace webrtc {

// static
rtc::scoped_refptr<EncodeUsageResource> EncodeUsageResource::Create(
    std::unique_ptr<OveruseFrameDetector> overuse_detector) {
  return rtc::make_ref_counted<EncodeUsageResource>(
      std::move(overuse_detector));
}

EncodeUsageResource::EncodeUsageResource(
    std::unique_ptr<OveruseFrameDetector> overuse_detector)
    : VideoStreamEncoderResource("EncoderUsageResource"),
      overuse_detector_(std::move(overuse_detector)),
      is_started_(false),
      target_frame_rate_(std::nullopt) {
  RTC_DCHECK(overuse_detector_);
}

EncodeUsageResource::~EncodeUsageResource() {}

bool EncodeUsageResource::is_started() const {
  RTC_DCHECK_RUN_ON(encoder_queue());
  return is_started_;
}

void EncodeUsageResource::StartCheckForOveruse(CpuOveruseOptions options) {
  RTC_DCHECK_RUN_ON(encoder_queue());
  RTC_DCHECK(!is_started_);
  overuse_detector_->StartCheckForOveruse(TaskQueueBase::Current(),
                                          std::move(options), this);
  is_started_ = true;
  overuse_detector_->OnTargetFramerateUpdated(TargetFrameRateAsInt());
}

void EncodeUsageResource::StopCheckForOveruse() {
  RTC_DCHECK_RUN_ON(encoder_queue());
  overuse_detector_->StopCheckForOveruse();
  is_started_ = false;
}

void EncodeUsageResource::SetTargetFrameRate(
    std::optional<double> target_frame_rate) {
  RTC_DCHECK_RUN_ON(encoder_queue());
  if (target_frame_rate == target_frame_rate_)
    return;
  target_frame_rate_ = target_frame_rate;
  if (is_started_)
    overuse_detector_->OnTargetFramerateUpdated(TargetFrameRateAsInt());
}

void EncodeUsageResource::OnEncodeStarted(const VideoFrame& cropped_frame,
                                          int64_t time_when_first_seen_us) {
  RTC_DCHECK_RUN_ON(encoder_queue());
  // TODO(hbos): Rename FrameCaptured() to something more appropriate (e.g.
  // "OnEncodeStarted"?) or revise usage.
  overuse_detector_->FrameCaptured(cropped_frame, time_when_first_seen_us);
}

void EncodeUsageResource::OnEncodeCompleted(
    uint32_t timestamp,
    int64_t time_sent_in_us,
    int64_t capture_time_us,
    std::optional<int> encode_duration_us) {
  RTC_DCHECK_RUN_ON(encoder_queue());
  // TODO(hbos): Rename FrameSent() to something more appropriate (e.g.
  // "OnEncodeCompleted"?).
  overuse_detector_->FrameSent(timestamp, time_sent_in_us, capture_time_us,
                               encode_duration_us);
}

void EncodeUsageResource::AdaptUp() {
  RTC_DCHECK_RUN_ON(encoder_queue());
  OnResourceUsageStateMeasured(ResourceUsageState::kUnderuse);
}

void EncodeUsageResource::AdaptDown() {
  RTC_DCHECK_RUN_ON(encoder_queue());
  OnResourceUsageStateMeasured(ResourceUsageState::kOveruse);
}

int EncodeUsageResource::TargetFrameRateAsInt() {
  RTC_DCHECK_RUN_ON(encoder_queue());
  return target_frame_rate_.has_value()
             ? static_cast<int>(target_frame_rate_.value())
             : std::numeric_limits<int>::max();
}

}  // namespace webrtc
