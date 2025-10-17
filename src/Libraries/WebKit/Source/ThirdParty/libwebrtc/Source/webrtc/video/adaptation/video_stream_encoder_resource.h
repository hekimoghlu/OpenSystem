/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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
#ifndef VIDEO_ADAPTATION_VIDEO_STREAM_ENCODER_RESOURCE_H_
#define VIDEO_ADAPTATION_VIDEO_STREAM_ENCODER_RESOURCE_H_

#include <optional>
#include <string>
#include <vector>

#include "api/adaptation/resource.h"
#include "api/sequence_checker.h"
#include "api/task_queue/task_queue_base.h"
#include "call/adaptation/adaptation_constraint.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {

class VideoStreamEncoderResource : public Resource {
 public:
  ~VideoStreamEncoderResource() override;

  // Registering task queues must be performed as part of initialization.
  void RegisterEncoderTaskQueue(TaskQueueBase* encoder_queue);

  // Resource implementation.
  std::string Name() const override;
  void SetResourceListener(ResourceListener* listener) override;

 protected:
  explicit VideoStreamEncoderResource(std::string name);

  void OnResourceUsageStateMeasured(ResourceUsageState usage_state);

  // The caller is responsible for ensuring the task queue is still valid.
  TaskQueueBase* encoder_queue() const;

 private:
  mutable Mutex lock_;
  const std::string name_;
  // Treated as const after initialization.
  TaskQueueBase* encoder_queue_;
  ResourceListener* listener_ RTC_GUARDED_BY(lock_);
};

}  // namespace webrtc

#endif  // VIDEO_ADAPTATION_VIDEO_STREAM_ENCODER_RESOURCE_H_
