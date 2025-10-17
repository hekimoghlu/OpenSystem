/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 27, 2024.
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

#include <wtf/Lock.h>
#include <wtf/Threading.h>

namespace WTF {

template<typename LockType>
class RecursiveLockAdapter {
public:
    RecursiveLockAdapter() = default;

    // Use WTF_IGNORES_THREAD_SAFETY_ANALYSIS because the function does conditional locking, which is
    // not supported by analysis. Also RecursiveLockAdapter may wrap a lock type besides WTF::Lock
    // which doesn't support analysis.
    void lock() WTF_IGNORES_THREAD_SAFETY_ANALYSIS
    {
        Thread& me = Thread::current();
        if (&me == m_owner) {
            m_recursionCount++;
            return;
        }
        
        m_lock.lock();
        ASSERT(!m_owner);
        ASSERT(!m_recursionCount);
        m_owner = &me;
        m_recursionCount = 1;
    }
    
    // Use WTF_IGNORES_THREAD_SAFETY_ANALYSIS because the function does conditional unlocking, which is
    // not supported by analysis. Also RecursiveLockAdapter may wrap a lock type besides WTF::Lock
    // which doesn't support analysis.
    void unlock() WTF_IGNORES_THREAD_SAFETY_ANALYSIS
    {
        if (--m_recursionCount)
            return;
        m_owner = nullptr;
        m_lock.unlock();
    }
    
    // Use WTF_IGNORES_THREAD_SAFETY_ANALYSIS because the function does conditional locking, which is
    // not supported by analysis. Also RecursiveLockAdapter may wrap a lock type besides WTF::Lock
    // which doesn't support analysis.
    bool tryLock() WTF_IGNORES_THREAD_SAFETY_ANALYSIS
    {
        Thread& me = Thread::current();
        if (&me == m_owner) {
            m_recursionCount++;
            return true;
        }
        
        if (!m_lock.tryLock())
            return false;
        
        ASSERT(!m_owner);
        ASSERT(!m_recursionCount);
        m_owner = &me;
        m_recursionCount = 1;
        return true;
    }
    
    bool isLocked() const
    {
        return m_lock.isLocked();
    }

    bool isOwner() const { return m_owner == &Thread::current(); }
    
private:
    Thread* m_owner { nullptr }; // Use Thread* instead of RefPtr<Thread> since m_owner thread is always alive while m_onwer is set.
    unsigned m_recursionCount { 0 };
    LockType m_lock;
};

using RecursiveLock = RecursiveLockAdapter<Lock>;

} // namespace WTF

using WTF::RecursiveLock;
