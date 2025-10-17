/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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

//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// WorkerThread_unittest:
//   Simple tests for the worker thread class.

#include <gtest/gtest.h>
#include <array>

#include "common/WorkerThread.h"

using namespace angle;

namespace
{

// Tests simple worker pool application.
TEST(WorkerPoolTest, SimpleTask)
{
    class TestTask : public Closure
    {
      public:
        void operator()() override { fired = true; }

        bool fired = false;
    };

    std::array<std::shared_ptr<WorkerThreadPool>, 2> pools = {
        {WorkerThreadPool::Create(1, ANGLEPlatformCurrent()),
         WorkerThreadPool::Create(0, ANGLEPlatformCurrent())}};
    for (auto &pool : pools)
    {
        std::array<std::shared_ptr<TestTask>, 4> tasks = {
            {std::make_shared<TestTask>(), std::make_shared<TestTask>(),
             std::make_shared<TestTask>(), std::make_shared<TestTask>()}};
        std::array<std::shared_ptr<WaitableEvent>, 4> waitables = {
            {pool->postWorkerTask(tasks[0]), pool->postWorkerTask(tasks[1]),
             pool->postWorkerTask(tasks[2]), pool->postWorkerTask(tasks[3])}};

        WaitableEvent::WaitMany(&waitables);

        for (const auto &task : tasks)
        {
            EXPECT_TRUE(task->fired);
        }
    }
}

}  // anonymous namespace
