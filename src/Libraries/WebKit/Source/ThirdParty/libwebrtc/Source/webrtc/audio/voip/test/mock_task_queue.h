/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 28, 2022.
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
#ifndef AUDIO_VOIP_TEST_MOCK_TASK_QUEUE_H_
#define AUDIO_VOIP_TEST_MOCK_TASK_QUEUE_H_

#include <memory>

#include "api/task_queue/task_queue_factory.h"
#include "api/task_queue/test/mock_task_queue_base.h"
#include "test/gmock.h"

namespace webrtc {

// MockTaskQueue enables immediate task run from global TaskQueueBase.
// It's necessary for some tests depending on TaskQueueBase internally.
class MockTaskQueue : public MockTaskQueueBase {
 public:
  MockTaskQueue() : current_(this) {}

  // Delete is deliberately defined as no-op as MockTaskQueue is expected to
  // hold onto current global TaskQueueBase throughout the testing.
  void Delete() override {}

 private:
  CurrentTaskQueueSetter current_;
};

class MockTaskQueueFactory : public TaskQueueFactory {
 public:
  explicit MockTaskQueueFactory(MockTaskQueue* task_queue)
      : task_queue_(task_queue) {}

  std::unique_ptr<TaskQueueBase, TaskQueueDeleter> CreateTaskQueue(
      absl::string_view /* name */,
      Priority /* priority */) const override {
    // Default MockTaskQueue::Delete is no-op, therefore it's safe to pass the
    // raw pointer.
    return std::unique_ptr<TaskQueueBase, TaskQueueDeleter>(task_queue_);
  }

 private:
  MockTaskQueue* task_queue_;
};

}  // namespace webrtc

#endif  // AUDIO_VOIP_TEST_MOCK_TASK_QUEUE_H_
