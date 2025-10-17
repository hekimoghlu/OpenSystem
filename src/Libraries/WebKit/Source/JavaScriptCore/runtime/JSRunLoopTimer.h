/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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

#include <wtf/HashSet.h>
#include <wtf/Lock.h>
#include <wtf/RefPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/RunLoop.h>
#include <wtf/SharedTask.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace JSC {

class JSLock;
class VM;

class JSRunLoopTimer : public ThreadSafeRefCounted<JSRunLoopTimer> {
public:
    typedef void TimerNotificationType();
    using TimerNotificationCallback = RefPtr<WTF::SharedTask<TimerNotificationType>>;

    class Manager {
        WTF_MAKE_NONCOPYABLE(Manager);
        WTF_MAKE_TZONE_ALLOCATED(Manager);
        void timerDidFireCallback();

        Manager() = default;

        void timerDidFire();

    public:
        static Manager& shared();
        void registerVM(VM&);
        void unregisterVM(VM&);
        void scheduleTimer(JSRunLoopTimer&, Seconds nextFireTime);
        void cancelTimer(JSRunLoopTimer&);

        std::optional<Seconds> timeUntilFire(JSRunLoopTimer&);

        // Do nothing since this is a singleton.
        void ref() const { }
        void deref() const { }

    private:
        Lock m_lock;

        class PerVMData {
            WTF_MAKE_FAST_ALLOCATED;
            WTF_MAKE_NONCOPYABLE(PerVMData);
        public:
            PerVMData(Manager&, WTF::RunLoop&);
            ~PerVMData();

            Ref<WTF::RunLoop> runLoop;
            std::unique_ptr<RunLoop::Timer> timer;
            Vector<std::pair<Ref<JSRunLoopTimer>, MonotonicTime>> timers;
        };

        UncheckedKeyHashMap<Ref<JSLock>, std::unique_ptr<PerVMData>> m_mapping;
    };

    JSRunLoopTimer(VM&);
    JS_EXPORT_PRIVATE virtual ~JSRunLoopTimer();
    virtual void doWork(VM&) = 0;

    JS_EXPORT_PRIVATE void setTimeUntilFire(Seconds intervalInSeconds);
    void cancelTimer();
    bool isScheduled() const { return m_isScheduled; }

    // Note: The only thing the timer notification callback cannot do is
    // call setTimeUntilFire(). This will cause a deadlock. It would not
    // be hard to make this work, however, there are no clients that need
    // this behavior. We should implement it only if we find that we need it.
    JS_EXPORT_PRIVATE void addTimerSetNotification(TimerNotificationCallback);
    JS_EXPORT_PRIVATE void removeTimerSetNotification(TimerNotificationCallback);

    JS_EXPORT_PRIVATE std::optional<Seconds> timeUntilFire();

protected:
    static constexpr Seconds s_decade { 60 * 60 * 24 * 365 * 10 };
    Ref<JSLock> m_apiLock;

private:
    friend class Manager;

    void timerDidFire();

    UncheckedKeyHashSet<TimerNotificationCallback> m_timerSetCallbacks;
    Lock m_timerCallbacksLock;

    Lock m_lock;
    bool m_isScheduled { false };
};
    
} // namespace JSC
