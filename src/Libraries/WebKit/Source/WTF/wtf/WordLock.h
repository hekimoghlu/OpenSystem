/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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

#include <wtf/Atomics.h>
#include <wtf/Compiler.h>
#include <wtf/Locker.h>
#include <wtf/Noncopyable.h>

namespace TestWebKitAPI {
struct LockInspector;
};

namespace WTF {

// A WordLock is a fully adaptive mutex that uses sizeof(void*) storage. It has a fast path that is
// similar to a spinlock, and a slow path that is similar to std::mutex. In most cases, you should use
// Lock instead. WordLock sits lower in the stack and is used to implement Lock, so Lock is the main
// client of WordLock.

// NOTE: This is also a great lock to use if you are very low in the stack. For example,
// PrintStream uses this so that ParkingLot and Lock can use PrintStream. This means that if you
// try to use dataLog to debug this code, you will have a bad time.

class WordLock final {
    WTF_MAKE_NONCOPYABLE(WordLock);
    WTF_MAKE_FAST_ALLOCATED;
public:
    constexpr WordLock() = default;

    void lock()
    {
        if (LIKELY(m_word.compareExchangeWeak(0, isLockedBit, std::memory_order_acquire))) {
            // WordLock acquired!
            return;
        }

        lockSlow();
    }

    void unlock()
    {
        if (LIKELY(m_word.compareExchangeWeak(isLockedBit, 0, std::memory_order_release))) {
            // WordLock released, and nobody was waiting!
            return;
        }

        unlockSlow();
    }

    bool isHeld() const
    {
        return m_word.load(std::memory_order_acquire) & isLockedBit;
    }

    bool isLocked() const
    {
        return isHeld();
    }

protected:
    friend struct TestWebKitAPI::LockInspector;
    
    static constexpr uintptr_t isLockedBit = 1;
    static constexpr uintptr_t isQueueLockedBit = 2;
    static constexpr uintptr_t queueHeadMask = 3;

    WTF_EXPORT_PRIVATE void lockSlow();
    WTF_EXPORT_PRIVATE void unlockSlow();

    // Method used for testing only.
    bool isFullyReset() const
    {
        return !m_word.load();
    }

    Atomic<uintptr_t> m_word { 0 };
};

} // namespace WTF

using WTF::WordLock;
