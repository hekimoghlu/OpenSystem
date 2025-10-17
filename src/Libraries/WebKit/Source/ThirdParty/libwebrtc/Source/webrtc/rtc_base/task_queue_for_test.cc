/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 15, 2023.
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
#include "rtc_base/task_queue_for_test.h"

#include <memory>
#include <utility>

#include "api/task_queue/default_task_queue_factory.h"
#include "api/task_queue/task_queue_base.h"

namespace webrtc {

TaskQueueForTest::TaskQueueForTest(
    std::unique_ptr<TaskQueueBase, TaskQueueDeleter> task_queue)
    : impl_(std::move(task_queue)) {}

TaskQueueForTest::TaskQueueForTest(absl::string_view name,
                                   TaskQueueFactory::Priority priority)
    : impl_(CreateDefaultTaskQueueFactory()->CreateTaskQueue(name, priority)) {}

TaskQueueForTest::~TaskQueueForTest() {
  // Stop the TaskQueue before invalidating impl_ pointer so that tasks that
  // race with the TaskQueueForTest destructor could still use TaskQueueForTest
  // functions like 'IsCurrent'.
  impl_.get_deleter()(impl_.get());
  impl_.release();
}

}  // namespace webrtc
