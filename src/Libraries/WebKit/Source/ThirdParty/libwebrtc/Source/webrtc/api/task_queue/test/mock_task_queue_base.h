/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 24, 2022.
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
#ifndef API_TASK_QUEUE_TEST_MOCK_TASK_QUEUE_BASE_H_
#define API_TASK_QUEUE_TEST_MOCK_TASK_QUEUE_BASE_H_

#include "absl/functional/any_invocable.h"
#include "api/task_queue/task_queue_base.h"
#include "api/units/time_delta.h"
#include "test/gmock.h"

namespace webrtc {

class MockTaskQueueBase : public TaskQueueBase {
 public:
  using TaskQueueBase::PostDelayedTaskTraits;
  using TaskQueueBase::PostTaskTraits;

  MOCK_METHOD(void, Delete, (), (override));
  MOCK_METHOD(void,
              PostTaskImpl,
              (absl::AnyInvocable<void() &&>,
               const PostTaskTraits&,
               const Location&),
              (override));
  MOCK_METHOD(void,
              PostDelayedTaskImpl,
              (absl::AnyInvocable<void() &&>,
               TimeDelta,
               const PostDelayedTaskTraits&,
               const Location&),
              (override));
};

}  // namespace webrtc

#endif  // API_TASK_QUEUE_TEST_MOCK_TASK_QUEUE_BASE_H_
