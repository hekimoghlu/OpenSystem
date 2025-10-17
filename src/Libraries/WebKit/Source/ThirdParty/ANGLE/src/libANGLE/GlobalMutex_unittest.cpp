/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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
// Copyright 2023 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// GlobalMutex_unittest:
//   Tests of the Scoped<*>GlobalMutexLock classes
//

#include <gtest/gtest.h>

#include "libANGLE/GlobalMutex.h"

namespace
{
template <class ScopedGlobalLockT, class... Args>
void runBasicGlobalMutexTest(bool expectToPass, Args &&...args)
{
    constexpr size_t kThreadCount    = 16;
    constexpr size_t kIterationCount = 50'000;

    std::array<std::thread, kThreadCount> threads;

    std::mutex mutex;
    std::condition_variable condVar;
    size_t readyCount = 0;

    std::atomic<size_t> testVar;

    for (size_t i = 0; i < kThreadCount; ++i)
    {
        threads[i] = std::thread([&]() {
            {
                std::unique_lock<std::mutex> lock(mutex);
                ++readyCount;
                if (readyCount < kThreadCount)
                {
                    condVar.wait(lock, [&]() { return readyCount == kThreadCount; });
                }
                else
                {
                    condVar.notify_all();
                }
            }
            for (size_t j = 0; j < kIterationCount; ++j)
            {
                ScopedGlobalLockT lock(std::forward<Args>(args)...);
                const int local    = testVar.load(std::memory_order_relaxed);
                const int newValue = local + 1;
                testVar.store(newValue, std::memory_order_relaxed);
            }
        });
    }

    for (size_t i = 0; i < kThreadCount; ++i)
    {
        threads[i].join();
    }

    if (expectToPass)
    {
        EXPECT_EQ(testVar.load(), kThreadCount * kIterationCount);
    }
    else
    {
        EXPECT_LE(testVar.load(), kThreadCount * kIterationCount);
    }
}

// Tests basic usage of ScopedGlobalEGLMutexLock.
TEST(GlobalMutexTest, ScopedGlobalEGLMutexLock)
{
    runBasicGlobalMutexTest<egl::ScopedGlobalEGLMutexLock>(true);
}

// Tests basic usage of ScopedOptionalGlobalMutexLock (Enabled).
TEST(GlobalMutexTest, ScopedOptionalGlobalMutexLockEnabled)
{
    runBasicGlobalMutexTest<egl::ScopedOptionalGlobalMutexLock>(true, true);
}

// Tests basic usage of ScopedOptionalGlobalMutexLock (Disabled).
TEST(GlobalMutexTest, ScopedOptionalGlobalMutexLockDisabled)
{
    runBasicGlobalMutexTest<egl::ScopedOptionalGlobalMutexLock>(false, false);
}

#if defined(ANGLE_ENABLE_GLOBAL_MUTEX_RECURSION)
// Tests that ScopedGlobalEGLMutexLock can be recursively locked.
TEST(GlobalMutexTest, RecursiveScopedGlobalEGLMutexLock)
{
    egl::ScopedGlobalEGLMutexLock lock;
    egl::ScopedGlobalEGLMutexLock lock2;
}

// Tests that ScopedOptionalGlobalMutexLock can be recursively locked.
TEST(GlobalMutexTest, RecursiveScopedOptionalGlobalMutexLock)
{
    egl::ScopedOptionalGlobalMutexLock lock(true);
    egl::ScopedOptionalGlobalMutexLock lock2(true);
}
#endif

}  // anonymous namespace
