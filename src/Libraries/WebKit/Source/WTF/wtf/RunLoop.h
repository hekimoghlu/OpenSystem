/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 14, 2022.
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
#include <wtf/CheckedPtr.h>
#include <wtf/Condition.h>
#include <wtf/Deque.h>
#include <wtf/Forward.h>
#include <wtf/FunctionDispatcher.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/Observer.h>
#include <wtf/RetainPtr.h>
#include <wtf/Seconds.h>
#include <wtf/ThreadSafetyAnalysis.h>
#include <wtf/ThreadSpecific.h>
#include <wtf/Threading.h>
#include <wtf/ThreadingPrimitives.h>
#include <wtf/TypeTraits.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/WTFString.h>

#if USE(CF)
#include <CoreFoundation/CFRunLoop.h>
#endif

#if USE(GLIB_EVENT_LOOP)
#include <wtf/glib/GRefPtr.h>
#endif

#if USE(GENERIC_EVENT_LOOP)
#include <wtf/RedBlackTree.h>
#endif

namespace WTF {

#if USE(COCOA_EVENT_LOOP)
class SchedulePair;
struct SchedulePairHash;
using SchedulePairHashSet = UncheckedKeyHashSet<RefPtr<SchedulePair>, SchedulePairHash>;
#endif

#if USE(CF)
using RunLoopMode = CFStringRef;
#define DefaultRunLoopMode kCFRunLoopDefaultMode
#else
using RunLoopMode = unsigned;
#define DefaultRunLoopMode 0
#endif

// Classes that offer Timers should be ref-counted of CanMakeCheckedPtr. Please do not add new exceptions.
template<typename T> struct IsDeprecatedTimerSmartPointerException : std::false_type { };

class WTF_CAPABILITY("is current") RunLoop final : public GuaranteedSerialFunctionDispatcher {
    WTF_MAKE_NONCOPYABLE(RunLoop);
public:
    // Must be called from the main thread.
    WTF_EXPORT_PRIVATE static void initializeMain();
#if USE(WEB_THREAD)
    WTF_EXPORT_PRIVATE static void initializeWeb();
#endif

    WTF_EXPORT_PRIVATE static RunLoop& current();
    static Ref<RunLoop> protectedCurrent() { return current(); }
    WTF_EXPORT_PRIVATE static RunLoop& main();
    static Ref<RunLoop> protectedMain() { return main(); }
#if USE(WEB_THREAD)
    WTF_EXPORT_PRIVATE static RunLoop& web();
    WTF_EXPORT_PRIVATE static RunLoop* webIfExists();
#endif
    WTF_EXPORT_PRIVATE static Ref<RunLoop> create(ASCIILiteral threadName, ThreadType = ThreadType::Unknown, Thread::QOS = Thread::QOS::UserInitiated);

    static bool isMain() { return main().isCurrent(); }
    WTF_EXPORT_PRIVATE bool isCurrent() const final;
    WTF_EXPORT_PRIVATE ~RunLoop() final;

    WTF_EXPORT_PRIVATE void dispatch(Function<void()>&&) final;
#if USE(COCOA_EVENT_LOOP)
    WTF_EXPORT_PRIVATE static void dispatch(const SchedulePairHashSet&, Function<void()>&&);
#endif

    WTF_EXPORT_PRIVATE static void run();
    WTF_EXPORT_PRIVATE void stop();
    WTF_EXPORT_PRIVATE void wakeUp();

    WTF_EXPORT_PRIVATE void suspendFunctionDispatchForCurrentCycle();

    enum class CycleResult { Continue, Stop };
    WTF_EXPORT_PRIVATE CycleResult static cycle(RunLoopMode = DefaultRunLoopMode);

    WTF_EXPORT_PRIVATE void threadWillExit();

#if USE(GLIB_EVENT_LOOP)
    WTF_EXPORT_PRIVATE GMainContext* mainContext() const { return m_mainContext.get(); }
    enum class Event { WillDispatch, DidDispatch };
    using Observer = WTF::Observer<void(Event, const String&)>;
    WTF_EXPORT_PRIVATE void observe(const Observer&);
#endif

#if USE(GENERIC_EVENT_LOOP) || USE(WINDOWS_EVENT_LOOP)
    WTF_EXPORT_PRIVATE static void setWakeUpCallback(WTF::Function<void()>&&);
#endif

#if USE(WINDOWS_EVENT_LOOP)
    static void registerRunLoopMessageWindowClass();
#endif

    class TimerBase {
        friend class RunLoop;
    public:
        WTF_EXPORT_PRIVATE explicit TimerBase(Ref<RunLoop>&&);
        WTF_EXPORT_PRIVATE virtual ~TimerBase();

        void startRepeating(Seconds interval) { start(std::max(interval, 0_s), true); }
        void startOneShot(Seconds interval) { start(std::max(interval, 0_s), false); }

        WTF_EXPORT_PRIVATE void stop();
        WTF_EXPORT_PRIVATE bool isActive() const;
        WTF_EXPORT_PRIVATE Seconds secondsUntilFire() const;

        virtual void fired() = 0;

#if USE(GLIB_EVENT_LOOP)
        WTF_EXPORT_PRIVATE void setName(ASCIILiteral);
        WTF_EXPORT_PRIVATE void setPriority(int);
#endif

    private:
        WTF_EXPORT_PRIVATE void start(Seconds interval, bool repeat);

        Ref<RunLoop> m_runLoop;

#if USE(WINDOWS_EVENT_LOOP)
        bool isActiveWithLock() const WTF_REQUIRES_LOCK(m_runLoop->m_loopLock);
        void timerFired();
        MonotonicTime m_nextFireDate;
        Seconds m_interval;
        bool m_isRepeating { false };
        bool m_isActive { false };
#elif USE(COCOA_EVENT_LOOP)
        RetainPtr<CFRunLoopTimerRef> m_timer;
#elif USE(GLIB_EVENT_LOOP)
        void updateReadyTime();
        GRefPtr<GSource> m_source;
        bool m_isRepeating { false };
        Seconds m_interval { 0 };
#elif USE(GENERIC_EVENT_LOOP)
        bool isActiveWithLock() const WTF_REQUIRES_LOCK(m_runLoop->m_loopLock);
        void stopWithLock() WTF_REQUIRES_LOCK(m_runLoop->m_loopLock);

        class ScheduledTask;
        Ref<ScheduledTask> m_scheduledTask;
#endif
    };

    class Timer : public TimerBase {
        WTF_MAKE_FAST_ALLOCATED;
    public:
        template <typename TimerFiredClass>
        requires (WTF::HasRefPtrMemberFunctions<TimerFiredClass>::value)
        Timer(Ref<RunLoop>&& runLoop, TimerFiredClass* object, void (TimerFiredClass::*function)())
            : Timer(WTFMove(runLoop), [object, function] {
                RefPtr protectedObject { object };
                (object->*function)();
            })
        {
        }

        template <typename TimerFiredClass>
        requires (WTF::HasCheckedPtrMemberFunctions<TimerFiredClass>::value && !WTF::HasRefPtrMemberFunctions<TimerFiredClass>::value)
        Timer(Ref<RunLoop>&& runLoop, TimerFiredClass* object, void (TimerFiredClass::*function)())
            : Timer(WTFMove(runLoop), [object, function] {
                CheckedPtr checkedObject { object };
                (object->*function)();
            })
        {
        }

        // FIXME: This constructor isn't as safe as the other ones and should be removed.
        template <typename TimerFiredClass>
        requires (!WTF::HasRefPtrMemberFunctions<TimerFiredClass>::value && !WTF::HasCheckedPtrMemberFunctions<TimerFiredClass>::value)
        Timer(Ref<RunLoop>&& runLoop, TimerFiredClass* object, void (TimerFiredClass::*function)())
            : Timer(WTFMove(runLoop), std::bind(function, object))
        {
            static_assert(IsDeprecatedTimerSmartPointerException<std::remove_cv_t<TimerFiredClass>>::value,
                "Classes that use Timer should be ref-counted or CanMakeCheckedPtr. Please do not add new exceptions."
            );
        }

        Timer(Ref<RunLoop>&& runLoop, Function<void ()>&& function)
            : TimerBase(WTFMove(runLoop))
            , m_function(WTFMove(function))
        {
        }

    private:
        void fired() override { m_function(); }

        Function<void()> m_function;
    };

    class DispatchTimer final : public TimerBase, public ThreadSafeRefCounted<DispatchTimer> {
        WTF_MAKE_FAST_ALLOCATED;
    public:
        DispatchTimer(RunLoop& runLoop)
            : TimerBase(runLoop)
        {
        }

        void setFunction(Function<void()>&& function)
        {
            m_function = WTFMove(function);
        }
    private:
        void fired() final { m_function(); }

        Function<void()> m_function;
    };

    WTF_EXPORT_PRIVATE Ref<RunLoop::DispatchTimer> dispatchAfter(Seconds, Function<void()>&&);

private:
    class Holder;
    static ThreadSpecific<Holder>& runLoopHolder();

    RunLoop();

    void performWork();

    Deque<Function<void()>> m_currentIteration;

    Lock m_nextIterationLock;
    Deque<Function<void()>> m_nextIteration WTF_GUARDED_BY_LOCK(m_nextIterationLock);

    bool m_isFunctionDispatchSuspended { false };
    bool m_hasSuspendedFunctions { false };

#if USE(WINDOWS_EVENT_LOOP)
    static LRESULT CALLBACK RunLoopWndProc(HWND, UINT, WPARAM, LPARAM);
    LRESULT wndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
    HWND m_runLoopMessageWindow;
    UncheckedKeyHashSet<UINT_PTR> m_liveTimers;

    Lock m_loopLock;
#elif USE(COCOA_EVENT_LOOP)
    static void performWork(void*);
    RetainPtr<CFRunLoopRef> m_runLoop;
    RetainPtr<CFRunLoopSourceRef> m_runLoopSource;
#elif USE(GLIB_EVENT_LOOP)
    void notify(Event, const char*);

    static GSourceFuncs s_runLoopSourceFunctions;
    GRefPtr<GMainContext> m_mainContext;
    Vector<GRefPtr<GMainLoop>> m_mainLoops;
    GRefPtr<GSource> m_source;
    WeakHashSet<Observer> m_observers;
#elif USE(GENERIC_EVENT_LOOP)
    void scheduleWithLock(TimerBase::ScheduledTask&) WTF_REQUIRES_LOCK(m_loopLock);
    void unscheduleWithLock(TimerBase::ScheduledTask&) WTF_REQUIRES_LOCK(m_loopLock);
    void wakeUpWithLock() WTF_REQUIRES_LOCK(m_loopLock);

    enum class RunMode {
        Iterate,
        Drain
    };

    enum class Status {
        Clear,
        Stopping,
    };
    void runImpl(RunMode);
    bool populateTasks(RunMode, Status&, Deque<Ref<TimerBase::ScheduledTask>>&);

    friend class TimerBase;

    Lock m_loopLock;
    Condition m_readyToRun;
    Condition m_stopCondition;
    RedBlackTree<TimerBase::ScheduledTask, MonotonicTime> m_schedules;
    Vector<Status*> m_mainLoops;
    bool m_shutdown { false };
    bool m_pendingTasks { false };
#endif

#if USE(GENERIC_EVENT_LOOP) || USE(WINDOWS_EVENT_LOOP)
    Function<void()> m_wakeUpCallback;
#endif
};

inline void assertIsCurrent(const RunLoop& runLoop) WTF_ASSERTS_ACQUIRED_CAPABILITY(runLoop)
{
    ASSERT_UNUSED(runLoop, runLoop.isCurrent());
}

} // namespace WTF

using WTF::RunLoop;
using WTF::RunLoopMode;
using WTF::assertIsCurrent;
