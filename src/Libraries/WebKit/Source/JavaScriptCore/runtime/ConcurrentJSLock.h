/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 18, 2023.
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

#include "DeferGC.h"
#include <wtf/Lock.h>
#include <wtf/NoLock.h>

namespace JSC {

using ConcurrentJSLock = Lock;
using ConcurrentJSLockerImpl = Locker<Lock>;

static_assert(sizeof(ConcurrentJSLock) == 1, "Regardless of status of concurrent JS flag, size of ConurrentJSLock is always one byte.");

class ConcurrentJSLockerBase : public AbstractLocker {
    WTF_MAKE_NONCOPYABLE(ConcurrentJSLockerBase);
public:
    explicit ConcurrentJSLockerBase(ConcurrentJSLock& lockable)
    {
        m_locker.emplace(lockable);
    }
    explicit ConcurrentJSLockerBase(ConcurrentJSLock* lockable)
    {
        if (lockable)
            m_locker.emplace(*lockable);
    }

    explicit ConcurrentJSLockerBase(NoLockingNecessaryTag)
    {
    }

    ~ConcurrentJSLockerBase()
    {
    }
    
    void unlockEarly() WTF_IGNORES_THREAD_SAFETY_ANALYSIS
    {
        if (m_locker)
            m_locker->unlockEarly();
    }

private:
    std::optional<ConcurrentJSLockerImpl> m_locker;
};

class GCSafeConcurrentJSLocker : public ConcurrentJSLockerBase {
public:
    GCSafeConcurrentJSLocker(ConcurrentJSLock& lockable, VM& vm)
        : ConcurrentJSLockerBase(lockable)
        , m_deferGC(vm)
    {
    }

    GCSafeConcurrentJSLocker(ConcurrentJSLock* lockable, VM& vm)
        : ConcurrentJSLockerBase(lockable)
        , m_deferGC(vm)
    {
    }

    ~GCSafeConcurrentJSLocker()
    {
        // We have to unlock early due to the destruction order of base
        // vs. derived classes. If we didn't, then we would destroy the 
        // DeferGC object before unlocking the lock which could cause a GC
        // and resulting deadlock.
        unlockEarly();
    }

private:
    DeferGC m_deferGC;
};

class ConcurrentJSLocker : public ConcurrentJSLockerBase {
public:
    ConcurrentJSLocker(ConcurrentJSLock& lockable)
        : ConcurrentJSLockerBase(lockable)
#if !defined(NDEBUG)
        , m_disallowGC(std::in_place)
#endif
    {
    }

    ConcurrentJSLocker(ConcurrentJSLock* lockable)
        : ConcurrentJSLockerBase(lockable)
#if !defined(NDEBUG)
        , m_disallowGC(std::in_place)
#endif
    {
    }

    ConcurrentJSLocker(NoLockingNecessaryTag)
        : ConcurrentJSLockerBase(NoLockingNecessary)
#if !defined(NDEBUG)
        , m_disallowGC(std::nullopt)
#endif
    {
    }
    
    ConcurrentJSLocker(int) = delete;

#if !defined(NDEBUG)
private:
    std::optional<DisallowGC> m_disallowGC;
#endif
};

} // namespace JSC
