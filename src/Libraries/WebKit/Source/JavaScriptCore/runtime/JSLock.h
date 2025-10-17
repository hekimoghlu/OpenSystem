/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 26, 2025.
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

#include <mutex>
#include <wtf/Assertions.h>
#include <wtf/ForbidHeapAllocation.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>
#include <wtf/Threading.h>
#include <wtf/text/AtomStringTable.h>

namespace JSC {

// To make it safe to use JavaScript on multiple threads, it is
// important to lock before doing anything that allocates a
// JavaScript data structure or that interacts with shared state
// such as the protect count hash table. The simplest way to lock
// is to create a local JSLockHolder object in the scope where the lock 
// must be held and pass it the context that requires protection. 
// The lock is recursive so nesting is ok. The JSLock 
// object also acts as a convenience short-hand for running important
// initialization routines.

// To avoid deadlock, sometimes it is necessary to temporarily
// release the lock. Since it is recursive you actually have to
// release all locks held by your thread. This is safe to do if
// you are executing code that doesn't require the lock, and you
// reacquire the right number of locks at the end. You can do this
// by constructing a locally scoped JSLock::DropAllLocks object. The 
// DropAllLocks object takes care to release the JSLock only if your
// thread acquired it to begin with.

class CallFrame;
class VM;
class JSGlobalObject;
class JSLock;

// FIXME: We should either have a specialization of WTF::Locker for JSLock or only allow using JSLockHolder.
// It's weird that WTF::Locker<JSLock> doesn't ref() the VM for the lifetime of the lock and it's unclear
// there's any noticable performance difference.
class JSLockHolder {
public:
    JS_EXPORT_PRIVATE JSLockHolder(VM*);
    JS_EXPORT_PRIVATE JSLockHolder(VM&);
    JS_EXPORT_PRIVATE JSLockHolder(JSGlobalObject*);

    JS_EXPORT_PRIVATE ~JSLockHolder();

private:
    RefPtr<VM> m_vm;
};

class JSLock : public ThreadSafeRefCounted<JSLock> {
    WTF_MAKE_NONCOPYABLE(JSLock);
public:
    JSLock(VM*);
    JS_EXPORT_PRIVATE ~JSLock();

    JS_EXPORT_PRIVATE void lock();
    JS_EXPORT_PRIVATE void unlock();

    static void lock(JSGlobalObject*);
    static void unlock(JSGlobalObject*);
    static void lock(VM&);
    static void unlock(VM&);

    VM* vm() { return m_vm; }

    std::optional<RefPtr<Thread>> ownerThread() const
    {
        if (m_hasOwnerThread)
            return m_ownerThread;
        return std::nullopt;
    }
    bool currentThreadIsHoldingLock() { return m_hasOwnerThread && m_ownerThread.get() == &Thread::current(); }

    void willDestroyVM(VM*);

    class DropAllLocks {
        WTF_MAKE_NONCOPYABLE(DropAllLocks);
    public:
        JS_EXPORT_PRIVATE DropAllLocks(JSGlobalObject*);
        JS_EXPORT_PRIVATE DropAllLocks(VM*);
        JS_EXPORT_PRIVATE DropAllLocks(VM&);
        JS_EXPORT_PRIVATE ~DropAllLocks();

        void setDropDepth(unsigned depth) { m_dropDepth = depth; }
        unsigned dropDepth() const { return m_dropDepth; }

    private:
        intptr_t m_droppedLockCount;
        RefPtr<VM> m_vm;
        unsigned m_dropDepth;
    };

    void makeWebThreadAware()
    {
        m_isWebThreadAware = true;
    }

    bool isWebThreadAware() const { return m_isWebThreadAware; }

private:
    void lock(intptr_t lockCount);
    void unlock(intptr_t unlockCount);

    void didAcquireLock();
    void willReleaseLock();

    unsigned dropAllLocks(DropAllLocks*);
    void grabAllLocks(DropAllLocks*, unsigned lockCount);

    Lock m_lock;
    bool m_isWebThreadAware { false };
    // We cannot make m_ownerThread an optional (instead of pairing it with an explicit
    // m_hasOwnerThread) because currentThreadIsHoldingLock() may be called from a
    // different thread, and an optional is vulnerable to races.
    // See https://bugs.webkit.org/show_bug.cgi?id=169042#c6
    bool m_hasOwnerThread { false };
    bool m_shouldReleaseHeapAccess;
    RefPtr<Thread> m_ownerThread;
    intptr_t m_lockCount;
    unsigned m_lockDropDepth;
    uint32_t m_lastOwnerThread { 0 };
    VM* m_vm;
    AtomStringTable* m_entryAtomStringTable; 
};

} // namespace
