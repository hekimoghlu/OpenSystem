/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#pragma once

#include "BAssert.h"
#include <atomic>
#include <mutex>
#include <thread>

// A fast replacement for std::mutex.

namespace bmalloc {

class Mutex;

using UniqueLockHolder = std::unique_lock<Mutex>;
using LockHolder = std::scoped_lock<Mutex>;

class Mutex {
public:
    constexpr Mutex() = default;

    void lock();
    bool try_lock();
    void unlock();

private:
    BEXPORT void lockSlowCase();

    std::atomic<bool> m_flag { false };
    std::atomic<bool> m_isSpinning { false };
};

static inline void sleep(
    UniqueLockHolder& lock, std::chrono::milliseconds duration)
{
    if (duration == std::chrono::milliseconds(0))
        return;
    
    lock.unlock();
    std::this_thread::sleep_for(duration);
    lock.lock();
}

static inline void waitUntilFalse(
    UniqueLockHolder& lock, std::chrono::milliseconds sleepDuration,
    bool& condition)
{
    while (condition) {
        condition = false;
        sleep(lock, sleepDuration);
    }
}

inline bool Mutex::try_lock()
{
    return !m_flag.exchange(true, std::memory_order_acquire);
}

inline void Mutex::lock()
{
    if (!try_lock())
        lockSlowCase();
}

inline void Mutex::unlock()
{
    m_flag.store(false, std::memory_order_release);
}

} // namespace bmalloc
