/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 30, 2025.
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
#include "video/adaptation/pixel_limit_resource.h"

#include "api/sequence_checker.h"
#include "api/units/time_delta.h"
#include "call/adaptation/video_stream_adapter.h"
#include "rtc_base/checks.h"

namespace webrtc {

namespace {

constexpr TimeDelta kResourceUsageCheckIntervalMs = TimeDelta::Seconds(5);

}  // namespace

// static
rtc::scoped_refptr<PixelLimitResource> PixelLimitResource::Create(
    TaskQueueBase* task_queue,
    VideoStreamInputStateProvider* input_state_provider) {
  return rtc::make_ref_counted<PixelLimitResource>(task_queue,
                                                   input_state_provider);
}

PixelLimitResource::PixelLimitResource(
    TaskQueueBase* task_queue,
    VideoStreamInputStateProvider* input_state_provider)
    : task_queue_(task_queue),
      input_state_provider_(input_state_provider),
      max_pixels_(std::nullopt) {
  RTC_DCHECK(task_queue_);
  RTC_DCHECK(input_state_provider_);
}

PixelLimitResource::~PixelLimitResource() {
  RTC_DCHECK(!listener_);
  RTC_DCHECK(!repeating_task_.Running());
}

void PixelLimitResource::SetMaxPixels(int max_pixels) {
  RTC_DCHECK_RUN_ON(task_queue_);
  max_pixels_ = max_pixels;
}

void PixelLimitResource::SetResourceListener(ResourceListener* listener) {
  RTC_DCHECK_RUN_ON(task_queue_);
  listener_ = listener;
  if (listener_) {
    repeating_task_.Stop();
    repeating_task_ = RepeatingTaskHandle::Start(task_queue_, [&] {
      RTC_DCHECK_RUN_ON(task_queue_);
      if (!listener_) {
        // We don't have a listener so resource adaptation must not be running,
        // try again later.
        return kResourceUsageCheckIntervalMs;
      }
      if (!max_pixels_.has_value()) {
        // No pixel limit configured yet, try again later.
        return kResourceUsageCheckIntervalMs;
      }
      std::optional<int> frame_size_pixels =
          input_state_provider_->InputState().frame_size_pixels();
      if (!frame_size_pixels.has_value()) {
        // We haven't observed a frame yet so we don't know if it's going to be
        // too big or too small, try again later.
        return kResourceUsageCheckIntervalMs;
      }
      int current_pixels = frame_size_pixels.value();
      int target_pixel_upper_bounds = max_pixels_.value();
      // To avoid toggling, we allow any resolutions between
      // `target_pixel_upper_bounds` and video_stream_adapter.h's
      // GetLowerResolutionThan(). This is the pixels we end up if we adapt down
      // from `target_pixel_upper_bounds`.
      int target_pixels_lower_bounds =
          GetLowerResolutionThan(target_pixel_upper_bounds);
      if (current_pixels > target_pixel_upper_bounds) {
        listener_->OnResourceUsageStateMeasured(
            rtc::scoped_refptr<Resource>(this), ResourceUsageState::kOveruse);
      } else if (current_pixels < target_pixels_lower_bounds) {
        listener_->OnResourceUsageStateMeasured(
            rtc::scoped_refptr<Resource>(this), ResourceUsageState::kUnderuse);
      }
      return kResourceUsageCheckIntervalMs;
    });
  } else {
    repeating_task_.Stop();
  }
  // The task must be running if we have a listener.
  RTC_DCHECK(repeating_task_.Running() || !listener_);
}

}  // namespace webrtc
