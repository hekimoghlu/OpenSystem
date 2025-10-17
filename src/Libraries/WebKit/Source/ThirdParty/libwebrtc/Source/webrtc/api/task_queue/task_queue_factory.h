/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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
#ifndef API_TASK_QUEUE_TASK_QUEUE_FACTORY_H_
#define API_TASK_QUEUE_TASK_QUEUE_FACTORY_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "api/task_queue/task_queue_base.h"

namespace webrtc {

// The implementation of this interface must be thread-safe.
class TaskQueueFactory {
 public:
  // TaskQueue priority levels. On some platforms these will map to thread
  // priorities, on others such as Mac and iOS, GCD queue priorities.
  enum class Priority { NORMAL = 0, HIGH, LOW };

  virtual ~TaskQueueFactory() = default;
  virtual std::unique_ptr<TaskQueueBase, TaskQueueDeleter> CreateTaskQueue(
      absl::string_view name,
      Priority priority) const = 0;
};

}  // namespace webrtc

#endif  // API_TASK_QUEUE_TASK_QUEUE_FACTORY_H_
