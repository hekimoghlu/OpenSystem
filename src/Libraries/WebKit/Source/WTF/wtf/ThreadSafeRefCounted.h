/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 6, 2022.
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

#include <atomic>
#include <wtf/FastMalloc.h>
#include <wtf/MainThread.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefCounted.h>

namespace WTF {

#if ASSERT_ENABLED || ENABLE(SECURITY_ASSERTIONS)
#define CHECK_THREAD_SAFE_REF_COUNTED_LIFECYCLE 1
#else
#define CHECK_THREAD_SAFE_REF_COUNTED_LIFECYCLE 0
#endif

class ThreadSafeRefCountedBase {
    WTF_MAKE_NONCOPYABLE(ThreadSafeRefCountedBase);
    WTF_MAKE_FAST_ALLOCATED;
public:
    ThreadSafeRefCountedBase() = default;

#if CHECK_THREAD_SAFE_REF_COUNTED_LIFECYCLE
    ~ThreadSafeRefCountedBase();
#endif

    void ref() const
    {
        applyRefDuringDestructionCheck();

        ++m_refCount;
    }

    bool hasOneRef() const
    {
        return refCount() == 1;
    }

    unsigned refCount() const
    {
        return m_refCount;
    }

protected:
    // Returns whether the pointer should be freed or not.
    bool derefBaseWithoutDeletionCheck() const
    {
        ASSERT(m_refCount);

        if (UNLIKELY(!--m_refCount)) {
            // Setting m_refCount to 1 here prevents double delete within the destructor but not from another thread
            // since such a thread could have ref'ed this object long after it had been deleted. See webkit.org/b/201576.
            m_refCount = 1;
#if CHECK_THREAD_SAFE_REF_COUNTED_LIFECYCLE
            m_deletionHasBegun = true;
#endif
            return true;
        }

        return false;
    }

    // Returns whether the pointer should be freed or not.
    bool derefBase() const
    {
        return derefBaseWithoutDeletionCheck();
    }

    void applyRefDuringDestructionCheck() const
    {
#if CHECK_THREAD_SAFE_REF_COUNTED_LIFECYCLE
        if (!m_deletionHasBegun)
            return;
        RefCountedBase::logRefDuringDestruction(this);
#endif
    }

private:
    mutable std::atomic<unsigned> m_refCount { 1 };

#if ASSERT_ENABLED
    // Match the layout of RefCounted, which has flag bits for threading checks.
    UNUSED_MEMBER_VARIABLE bool m_unused1;
    UNUSED_MEMBER_VARIABLE bool m_unused2;
#endif

#if CHECK_THREAD_SAFE_REF_COUNTED_LIFECYCLE
    mutable std::atomic<bool> m_deletionHasBegun { false };
    // Match the layout of RefCounted.
    UNUSED_MEMBER_VARIABLE bool m_unused3;
#endif
};

#if CHECK_THREAD_SAFE_REF_COUNTED_LIFECYCLE
inline ThreadSafeRefCountedBase::~ThreadSafeRefCountedBase()
{
    // When this ThreadSafeRefCounted object is a part of another object, derefBase() is never called on this object.
    m_deletionHasBegun = true;

    // FIXME: Test performance, then add a RELEASE_ASSERT for this too.
    if (m_refCount != 1)
        RefCountedBase::printRefDuringDestructionLogAndCrash(this);
}
#endif

template<class T, DestructionThread destructionThread = DestructionThread::Any> class ThreadSafeRefCounted : public ThreadSafeRefCountedBase {
public:
    void deref() const
    {
        if (!derefBase())
            return;

        if constexpr (destructionThread == DestructionThread::Any) {
            delete static_cast<const T*>(this);
        } else if constexpr (destructionThread == DestructionThread::Main) {
            ensureOnMainThread([this] {
                delete static_cast<const T*>(this);
            });
        } else if constexpr (destructionThread == DestructionThread::MainRunLoop) {
            ensureOnMainRunLoop([this] {
                delete static_cast<const T*>(this);
            });
        } else {
            STATIC_ASSERT_NOT_REACHED_FOR_VALUE(destructionThread, "Unexpected destructionThread enumerator value");
        }
    }

protected:
    ThreadSafeRefCounted() = default;
};

} // namespace WTF

using WTF::ThreadSafeRefCounted;
