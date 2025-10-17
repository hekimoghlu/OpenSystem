/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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

#include <functional>
#include <wtf/Atomics.h>
#include <wtf/Lock.h>
#include <wtf/MainThread.h>
#include <wtf/RunLoop.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

template <typename T>
class MainThreadNotifier final : public ThreadSafeRefCounted<MainThreadNotifier<T>> {
public:
    static Ref<MainThreadNotifier> create()
    {
        return adoptRef(*new MainThreadNotifier());
    }

    ~MainThreadNotifier()
    {
        ASSERT(!m_isValid.load());
    }

    bool isValid() const { return m_isValid.load(); }

    template<typename F>
    void notify(T notificationType, F&& callbackFunctor)
    {
        ASSERT(m_isValid.load());
        // Assert that there is only one bit on at a time.
        ASSERT(!(static_cast<int>(notificationType) & (static_cast<int>(notificationType) - 1)));
        if (isMainThread()) {
            removePendingNotification(notificationType);
            callbackFunctor();
            return;
        }

        if (!addPendingNotification(notificationType))
            return;

        RunLoop::main().dispatch([this, protectedThis = Ref { *this }, notificationType, callback = Function<void()>(WTFMove(callbackFunctor))] {
            if (!m_isValid.load())
                return;
            if (removePendingNotification(notificationType))
                callback();
        });
    }

    template<typename F>
    void notifyAndWait(T notificationType, F&& callbackFunctor)
    {
        Lock lock;
        Condition condition;

        notify(notificationType, [functor = WTFMove(callbackFunctor), &condition, &lock] {
            functor();
            Locker locker { lock };
            condition.notifyOne();
        });

        if (!isMainThread()) {
            Locker locker { lock };
            condition.wait(lock);
        }
    }

    void cancelPendingNotifications(unsigned mask = 0)
    {
        ASSERT(m_isValid.load());
        Locker locker { m_pendingNotificationsLock };
        if (mask)
            m_pendingNotifications &= ~mask;
        else
            m_pendingNotifications = 0;
    }

    void invalidate()
    {
        ASSERT(m_isValid.load());
        m_isValid.store(false);
    }

private:
    MainThreadNotifier()
    {
        m_isValid.store(true);
    }

    bool addPendingNotification(T notificationType)
    {
        Locker locker { m_pendingNotificationsLock };
        if (notificationType & m_pendingNotifications)
            return false;
        m_pendingNotifications |= notificationType;
        return true;
    }

    bool removePendingNotification(T notificationType)
    {
        Locker locker { m_pendingNotificationsLock };
        if (notificationType & m_pendingNotifications) {
            m_pendingNotifications &= ~notificationType;
            return true;
        }
        return false;
    }

    Lock m_pendingNotificationsLock;
    unsigned m_pendingNotifications WTF_GUARDED_BY_LOCK(m_pendingNotificationsLock) { 0 };
    Atomic<bool> m_isValid;
};

} // namespace WebCore

