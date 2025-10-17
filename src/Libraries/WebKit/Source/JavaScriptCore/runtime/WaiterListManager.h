/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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

#include "DeferredWorkTimer.h"
#include "JSPromise.h"
#include <wtf/Condition.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/SentinelLinkedList.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

enum class AtomicsWaitType : uint8_t { Sync, Async };
enum class AtomicsWaitValidation : uint8_t { Pass, Fail };

class Waiter final : public WTF::BasicRawSentinelNode<Waiter>, public ThreadSafeRefCounted<Waiter> {
    WTF_MAKE_TZONE_ALLOCATED(Waiter);

public:
    Waiter(VM*);
    Waiter(JSPromise*);

    bool isAsync() const
    {
        return m_isAsync;
    }

    VM* vm()
    {
        return m_vm;
    }

    Condition& condition()
    {
        ASSERT(!m_isAsync);
        return m_condition;
    }

    RefPtr<DeferredWorkTimer::TicketData> ticket(const AbstractLocker&) const
    {
        ASSERT(m_isAsync);
        return m_ticket.get();
    }

    void clearTicket(const AbstractLocker&)
    {
        ASSERT(m_isAsync);
        m_ticket = nullptr;
    }

    void setTimer(const AbstractLocker&, Ref<RunLoop::DispatchTimer>&& timer)
    {
        ASSERT(m_isAsync);
        m_timer = WTFMove(timer);
    }

    bool hasTimer(const AbstractLocker&)
    {
        return !!m_timer;
    }

    void clearTimer(const AbstractLocker&)
    {
        ASSERT(m_isAsync);
        // If the timeout for AsyncWaiter is infinity, we won't dispatch any timer.
        if (!m_timer)
            return;
        m_timer->stop();
        // The AsyncWaiter's timer holds the waiter's reference. This
        // releases the strong reference to the Waiter in the timer.
        m_timer = nullptr;
    }

    void scheduleWorkAndClear(const AbstractLocker&, DeferredWorkTimer::Task&&);
    void cancelAndClear(const AbstractLocker&);
    void dump(PrintStream&) const;

private:
    VM* m_vm { nullptr };
    ThreadSafeWeakPtr<DeferredWorkTimer::TicketData> m_ticket { nullptr };
    RefPtr<RunLoop::DispatchTimer> m_timer { nullptr };
    Condition m_condition;
    bool m_isAsync { false };
};

class WaiterList : public ThreadSafeRefCounted<WaiterList> {
    WTF_MAKE_TZONE_ALLOCATED(WaiterList);

public:
    ~WaiterList()
    {
        removeIf([](Waiter*) {
            return true;
        });
    }

    void addLast(const AbstractLocker&, Waiter& waiter)
    {
        m_waiters.append(&waiter);
        waiter.ref();
        m_size++;
    }

    Ref<Waiter> takeFirst(const AbstractLocker&)
    {
        // `takeFisrt` is used to consume a waiter (either notify, timeout, or remove).
        // So, the waiter must not be removed and belong to this list.
        Waiter& waiter = *m_waiters.begin();
        ASSERT((!waiter.isAsync() || waiter.ticket(NoLockingNecessary)) && waiter.vm() && waiter.isOnList());
        Ref<Waiter> protectedWaiter = Ref { waiter };
        removeWithUpdate(waiter);
        return protectedWaiter;
    }

    bool findAndRemove(const AbstractLocker&, Waiter& target)
    {
#if ASSERT_ENABLED
        if (target.isOnList()) {
            bool found = false;
            for (auto iter = m_waiters.begin(); iter != m_waiters.end(); ++iter) {
                if (&*iter == &target)
                    found = true;
            }
            ASSERT(found);
        }
#endif

        if (!target.isOnList())
            return false;
        removeWithUpdate(target);
        return true;
    }

    template<typename Functor>
    void removeIf(const AbstractLocker&, const Functor& functor)
    {
        removeIf(functor);
    }

    unsigned size()
    {
        return m_size;
    }

    Lock lock;

private:
    template<typename Functor>
    void removeIf(const Functor& functor)
    {
        m_waiters.forEach([&](Waiter* waiter) {
            if (functor(waiter))
                removeWithUpdate(*waiter);
        });
    }

    void removeWithUpdate(Waiter& waiter)
    {
        m_waiters.remove(&waiter);
        waiter.deref();
        m_size--;
    }

    unsigned m_size { 0 };
    SentinelLinkedList<Waiter, BasicRawSentinelNode<Waiter>> m_waiters;
};

class WaiterListManager {
    WTF_MAKE_TZONE_ALLOCATED(WaiterListManager);
public:
    static WaiterListManager& singleton();

    enum class WaitSyncResult : int32_t {
        OK = 0,
        NotEqual = 1,
        TimedOut = 2,
        Terminated = 3,
    };

    JS_EXPORT_PRIVATE WaitSyncResult waitSync(VM&, int32_t* ptr, int32_t expected, Seconds timeout);
    JS_EXPORT_PRIVATE WaitSyncResult waitSync(VM&, int64_t* ptr, int64_t expected, Seconds timeout);
    JS_EXPORT_PRIVATE JSValue waitAsync(JSGlobalObject*, VM&, int32_t* ptr, int32_t expected, Seconds timeout);
    JS_EXPORT_PRIVATE JSValue waitAsync(JSGlobalObject*, VM&, int64_t* ptr, int64_t expected, Seconds timeout);

    enum class ResolveResult : uint8_t { Ok, Timeout };
    unsigned notifyWaiter(void* ptr, unsigned count);

    size_t waiterListSize(void* ptr);

    size_t totalWaiterCount();

    void unregister(VM*);
    void unregister(JSGlobalObject*);
    void unregister(uint8_t* arrayPtr, size_t);

private:
    template <typename ValueType>
    WaitSyncResult waitSyncImpl(VM&, ValueType* ptr, ValueType expectedValue, Seconds timeout);
    template <typename ValueType>
    JSValue waitAsyncImpl(JSGlobalObject*, VM&, ValueType* ptr, ValueType expectedValue, Seconds timeout);

    // Notify the waiter if its ticket is not canceled.
    void notifyWaiterImpl(const AbstractLocker&, Ref<Waiter>&&, const ResolveResult);

    void timeoutAsyncWaiter(void* ptr, Ref<Waiter>&&);

    void cancelAsyncWaiter(const AbstractLocker&, Waiter*);

    Ref<WaiterList> findOrCreateList(void* ptr);

    RefPtr<WaiterList> findList(void* ptr);

    Lock m_waiterListsLock;
    UncheckedKeyHashMap<void*, Ref<WaiterList>> m_waiterLists;
};

} // namespace JSC
