/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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

#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#include <mutex>
#include <wtf/Assertions.h>
#include <wtf/Atomics.h>
#include <wtf/Compiler.h>
#include <wtf/ForbidHeapAllocation.h>
#include <wtf/Noncopyable.h>
#include <wtf/ThreadSanitizerSupport.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

namespace WTF {

enum NoLockingNecessaryTag { NoLockingNecessary };

class AbstractLocker {
    WTF_MAKE_NONCOPYABLE(AbstractLocker);
public:
    AbstractLocker(NoLockingNecessaryTag)
    {
    }
    
protected:
    AbstractLocker()
    {
    }
};

template<typename T> class DropLockForScope;

using AdoptLockTag = std::adopt_lock_t;
constexpr AdoptLockTag AdoptLock;

template<typename T, typename = void>
class [[nodiscard]] Locker : public AbstractLocker { // NOLINT
public:
    explicit Locker(T& lockable) : m_lockable(&lockable) { lock(); }
    explicit Locker(T* lockable) : m_lockable(lockable) { lock(); }
    explicit Locker(AdoptLockTag, T& lockable) : m_lockable(&lockable) { }

    // You should be wary of using this constructor. It's only applicable
    // in places where there is a locking protocol for a particular object
    // but it's not necessary to engage in that protocol yet. For example,
    // this often happens when an object is newly allocated and it can not
    // be accessed concurrently.
    Locker(NoLockingNecessaryTag) : m_lockable(nullptr) { }
    
    Locker(std::underlying_type_t<NoLockingNecessaryTag>) = delete;

    ~Locker()
    {
        unlock();
    }
    
    static Locker tryLock(T& lockable)
    {
        Locker result(NoLockingNecessary);
        if (lockable.tryLock()) {
            result.m_lockable = &lockable;
            return result;
        }
        return result;
    }

    T* lockable() { return m_lockable; }
    
    explicit operator bool() const { return !!m_lockable; }
    
    void unlockEarly()
    {
        unlock();
        m_lockable = nullptr;
    }
    
    // It's great to be able to pass lockers around. It enables custom locking adaptors like
    // JSC::LockDuringMarking.
    Locker(Locker&& other)
        : m_lockable(other.m_lockable)
    {
        ASSERT(&other != this);
        other.m_lockable = nullptr;
    }
    
    Locker& operator=(Locker&& other)
    {
        ASSERT(&other != this);
        m_lockable = other.m_lockable;
        other.m_lockable = nullptr;
        return *this;
    }
    
private:
    template<typename>
    friend class DropLockForScope;

    void unlock()
    {
        compilerFence();
        if (m_lockable) {
            TSAN_ANNOTATE_HAPPENS_BEFORE(m_lockable);
            m_lockable->unlock();
        }
    }

    void lock()
    {
        if (m_lockable) {
            m_lockable->lock();
            TSAN_ANNOTATE_HAPPENS_AFTER(m_lockable);
        }
        compilerFence();
    }
    
    T* m_lockable;
};

template<typename LockType>
class DropLockForScope : private AbstractLocker {
    WTF_FORBID_HEAP_ALLOCATION(DropLockForScope);
public:
    DropLockForScope(Locker<LockType>& lock)
        : m_lock(lock)
    {
        m_lock.unlock();
    }

    ~DropLockForScope()
    {
        m_lock.lock();
    }

private:
    Locker<LockType>& m_lock;
};

// This is a close replica of Locker, but for generic lock/unlock functions.
template<typename T, void (lockFunction)(T*), void (*unlockFunction)(T*)>
class ExternalLocker: public WTF::AbstractLocker {
public:
    explicit ExternalLocker(T* lockable)
        : m_lockable(lockable)
    {
        ASSERT(lockable);
        lock();
    }

    ~ExternalLocker()
    {
        unlock();
    }

    T* lockable() { return m_lockable; }

    explicit operator bool() const { return !!m_lockable; }

    void unlockEarly()
    {
        unlock();
        m_lockable = nullptr;
    }

    ExternalLocker(ExternalLocker&& other)
        : m_lockable(other.m_lockable)
    {
        ASSERT(&other != this);
        other.m_lockable = nullptr;
    }

    ExternalLocker& operator=(ExternalLocker&& other)
    {
        ASSERT(&other != this);
        m_lockable = other.m_lockable;
        other.m_lockable = nullptr;
        return *this;
    }

private:
    template<typename>
    friend class DropLockForScope;

    void unlock()
    {
        if (m_lockable)
            unlockFunction(m_lockable);
    }

    void lock()
    {
        if (m_lockable)
            lockFunction(m_lockable);
    }

    T* m_lockable;
};

}

using WTF::AbstractLocker;
using WTF::AdoptLock;
using WTF::Locker;
using WTF::NoLockingNecessaryTag;
using WTF::NoLockingNecessary;
using WTF::DropLockForScope;
using WTF::ExternalLocker;
