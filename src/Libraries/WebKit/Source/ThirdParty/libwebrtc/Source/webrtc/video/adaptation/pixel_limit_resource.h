/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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
#ifndef VIDEO_ADAPTATION_PIXEL_LIMIT_RESOURCE_H_
#define VIDEO_ADAPTATION_PIXEL_LIMIT_RESOURCE_H_

#include <optional>
#include <string>

#include "api/adaptation/resource.h"
#include "api/scoped_refptr.h"
#include "call/adaptation/video_stream_input_state_provider.h"
#include "rtc_base/task_utils/repeating_task.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

// An adaptation resource designed to be used in the TestBed. Used to simulate
// being CPU limited.
//
// Periodically reports "overuse" or "underuse" (every 5 seconds) until the
// stream is within the bounds specified in terms of a maximum resolution and
// one resolution step lower than that (this avoids toggling when this is the
// only resource in play). When multiple resources come in to play some amount
// of toggling is still possible in edge cases but that is OK for testing
// purposes.
class PixelLimitResource : public Resource {
 public:
  static rtc::scoped_refptr<PixelLimitResource> Create(
      TaskQueueBase* task_queue,
      VideoStreamInputStateProvider* input_state_provider);

  PixelLimitResource(TaskQueueBase* task_queue,
                     VideoStreamInputStateProvider* input_state_provider);
  ~PixelLimitResource() override;

  void SetMaxPixels(int max_pixels);

  // Resource implementation.
  std::string Name() const override { return "PixelLimitResource"; }
  void SetResourceListener(ResourceListener* listener) override;

 private:
  TaskQueueBase* const task_queue_;
  VideoStreamInputStateProvider* const input_state_provider_;
  std::optional<int> max_pixels_ RTC_GUARDED_BY(task_queue_);
  webrtc::ResourceListener* listener_ RTC_GUARDED_BY(task_queue_);
  RepeatingTaskHandle repeating_task_ RTC_GUARDED_BY(task_queue_);
};

}  // namespace webrtc

#endif  // VIDEO_ADAPTATION_PIXEL_LIMIT_RESOURCE_H_
