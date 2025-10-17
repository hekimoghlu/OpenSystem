/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 15, 2022.
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
#ifndef API_TASK_QUEUE_TASK_QUEUE_TEST_H_
#define API_TASK_QUEUE_TASK_QUEUE_TEST_H_

#include <functional>
#include <memory>

#include "api/field_trials_view.h"
#include "api/task_queue/task_queue_factory.h"
#include "test/gtest.h"

namespace webrtc {

// Suite of tests to verify TaskQueue implementation with.
// Example usage:
//
// namespace {
//
// using ::testing::Values;
// using ::webrtc::TaskQueueTest;
//
// std::unique_ptr<webrtc::TaskQueueFactory> CreateMyFactory();
//
// INSTANTIATE_TEST_SUITE_P(My, TaskQueueTest, Values(CreateMyFactory));
//
// }  // namespace
class TaskQueueTest
    : public ::testing::TestWithParam<std::function<
          std::unique_ptr<TaskQueueFactory>(const FieldTrialsView*)>> {};

}  // namespace webrtc

#endif  // API_TASK_QUEUE_TASK_QUEUE_TEST_H_
