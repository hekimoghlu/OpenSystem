/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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

namespace WTF {

// This is the algorithm used by WTF::Lock. You can use it to project one lock onto any atomic
// field. The limit of one lock is due to the use of the field's address as a key to find the lock's
// queue.

template<typename LockType>
struct EmptyLockHooks {
    static LockType lockHook(LockType value) { return value; }
    static LockType unlockHook(LockType value) { return value; }
    static LockType parkHook(LockType value) { return value; }
    static LockType handoffHook(LockType value) { return value; }
};

template<typename LockType, LockType isHeldBit, LockType hasParkedBit, typename Hooks = EmptyLockHooks<LockType>>
class LockAlgorithm {
    static constexpr bool verbose = false;
    static constexpr LockType mask = isHeldBit | hasParkedBit;

public:
    static bool lockFastAssumingZero(Atomic<LockType>& lock)
    {
        return lock.compareExchangeWeak(0, Hooks::lockHook(isHeldBit), std::memory_order_acquire);
    }
    
    static bool lockFast(Atomic<LockType>& lock)
    {
        return lock.transaction(
            [&] (LockType& value) -> bool {
                if (value & isHeldBit)
                    return false;
                value |= isHeldBit;
                value = Hooks::lockHook(value);
                return true;
            },
            std::memory_order_acquire);
    }
    
    static void lock(Atomic<LockType>& lock)
    {
        if (UNLIKELY(!lockFast(lock)))
            lockSlow(lock);
    }
    
    static bool tryLock(Atomic<LockType>& lock)
    {
        for (;;) {
            LockType currentValue = lock.load(std::memory_order_relaxed);
            if (currentValue & isHeldBit)
                return false;
            if (lock.compareExchangeWeak(currentValue, Hooks::lockHook(currentValue | isHeldBit), std::memory_order_acquire))
                return true;
        }
    }

    static bool unlockFastAssumingZero(Atomic<LockType>& lock)
    {
        return lock.compareExchangeWeak(isHeldBit, Hooks::unlockHook(0), std::memory_order_release);
    }
    
    static bool unlockFast(Atomic<LockType>& lock)
    {
        return lock.transaction(
            [&] (LockType& value) -> bool {
                if ((value & mask) != isHeldBit)
                    return false;
                value &= ~isHeldBit;
                value = Hooks::unlockHook(value);
                return true;
            },
            std::memory_order_release);
    }
    
    static void unlock(Atomic<LockType>& lock)
    {
        if (UNLIKELY(!unlockFast(lock)))
            unlockSlow(lock, Unfair);
    }
    
    static void unlockFairly(Atomic<LockType>& lock)
    {
        if (UNLIKELY(!unlockFast(lock)))
            unlockSlow(lock, Fair);
    }
    
    static bool safepointFast(const Atomic<LockType>& lock)
    {
        WTF::compilerFence();
        return !(lock.load(std::memory_order_relaxed) & hasParkedBit);
    }
    
    static void safepoint(Atomic<LockType>& lock)
    {
        if (UNLIKELY(!safepointFast(lock)))
            safepointSlow(lock);
    }
    
    static bool isLocked(const Atomic<LockType>& lock)
    {
        return lock.load(std::memory_order_acquire) & isHeldBit;
    }
    
    NEVER_INLINE static void lockSlow(Atomic<LockType>& lock);
    
    enum Fairness {
        Unfair,
        Fair
    };
    NEVER_INLINE static void unlockSlow(Atomic<LockType>& lock, Fairness fairness = Unfair);
    
    NEVER_INLINE static void safepointSlow(Atomic<LockType>& lockWord)
    {
        unlockFairly(lockWord);
        lock(lockWord);
    }
    
private:
    enum Token {
        BargingOpportunity,
        DirectHandoff
    };
};

} // namespace WTF

using WTF::LockAlgorithm;
