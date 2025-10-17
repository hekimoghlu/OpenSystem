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
#include "config.h"
#include <wtf/RunLoop.h>

#include <wtf/WindowsExtras.h>

namespace WTF {

static const UINT PerformWorkMessage = WM_USER + 1;
static const UINT SetTimerMessage = WM_USER + 2;
static const UINT KillTimerMessage = WM_USER + 3;
static const LPCWSTR kRunLoopMessageWindowClassName = L"RunLoopMessageWindow";

LRESULT CALLBACK RunLoop::RunLoopWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    if (RunLoop* runLoop = static_cast<RunLoop*>(getWindowPointer(hWnd, 0)))
        return runLoop->wndProc(hWnd, message, wParam, lParam);

    if (message == WM_CREATE) {
        LPCREATESTRUCT createStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);

        // Associate the RunLoop with the window.
        setWindowPointer(hWnd, 0, createStruct->lpCreateParams);
        return 0;
    }

    return ::DefWindowProc(hWnd, message, wParam, lParam);
}

LRESULT RunLoop::wndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message) {
    case PerformWorkMessage:
        performWork();
        return 0;
    case SetTimerMessage:
        ::SetTimer(hWnd, wParam, lParam, nullptr);
        return 0;
    case KillTimerMessage:
        ::KillTimer(hWnd, wParam);
        return 0;
    case WM_TIMER:
        RunLoop::TimerBase* timer = nullptr;
        {
            Locker locker { m_loopLock };
            if (m_liveTimers.contains(wParam))
                timer = std::bit_cast<RunLoop::TimerBase*>(wParam);
        }
        if (timer != nullptr)
            timer->timerFired();
        return 0;
    }

    return ::DefWindowProc(hWnd, message, wParam, lParam);
}

void RunLoop::run()
{
    MSG message;
    while (BOOL result = ::GetMessage(&message, nullptr, 0, 0)) {
        if (result == -1)
            break;
        ::TranslateMessage(&message);
        ::DispatchMessage(&message);
    }
}

void RunLoop::setWakeUpCallback(WTF::Function<void()>&& function)
{
    RunLoop::current().m_wakeUpCallback = WTFMove(function);
}

void RunLoop::stop()
{
    // RunLoop::stop() can be called from threads unrelated to this RunLoop.
    // We should post a message that call PostQuitMessage in RunLoop's thread.
    dispatch([] {
        ::PostQuitMessage(0);
    });
}

void RunLoop::registerRunLoopMessageWindowClass()
{
    WNDCLASS windowClass = { };
    windowClass.lpfnWndProc = RunLoop::RunLoopWndProc;
    windowClass.cbWndExtra = sizeof(RunLoop*);
    windowClass.lpszClassName = kRunLoopMessageWindowClassName;
    bool result = ::RegisterClass(&windowClass);
    RELEASE_ASSERT(result);
}

RunLoop::RunLoop()
{
    m_runLoopMessageWindow = ::CreateWindow(kRunLoopMessageWindowClassName, nullptr, 0,
        CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, HWND_MESSAGE, nullptr, nullptr, this);
    RELEASE_ASSERT(::IsWindow(m_runLoopMessageWindow));
}

RunLoop::~RunLoop()
{
    ::DestroyWindow(m_runLoopMessageWindow);
}

void RunLoop::wakeUp()
{
    // FIXME: No need to wake up the run loop if we've already called dispatch
    // before the run loop has had the time to respond.
    ::PostMessage(m_runLoopMessageWindow, PerformWorkMessage, reinterpret_cast<WPARAM>(this), 0);

    if (m_wakeUpCallback)
        m_wakeUpCallback();
}

RunLoop::CycleResult RunLoop::cycle(RunLoopMode)
{
    MSG message;
    while (::PeekMessage(&message, nullptr, 0, 0, PM_REMOVE)) {
        if (message.message == WM_QUIT)
            return CycleResult::Stop;

        ::TranslateMessage(&message);
        ::DispatchMessage(&message);
    }
    return CycleResult::Continue;
}

// RunLoop::Timer

void RunLoop::TimerBase::timerFired()
{
    {
        Locker locker { m_runLoop->m_loopLock };

        if (!m_isActive)
            return;

        if (!m_isRepeating) {
            m_isActive = false;
            ::KillTimer(m_runLoop->m_runLoopMessageWindow, std::bit_cast<uintptr_t>(this));
        } else
            m_nextFireDate = MonotonicTime::timePointFromNow(m_interval);
    }

    fired();
}

RunLoop::TimerBase::TimerBase(Ref<RunLoop>&& runLoop)
    : m_runLoop(WTFMove(runLoop))
{
}

RunLoop::TimerBase::~TimerBase()
{
    stop();
}

void RunLoop::TimerBase::start(Seconds interval, bool repeat)
{
    Locker locker { m_runLoop->m_loopLock };
    m_isRepeating = repeat;
    m_isActive = true;
    m_interval = interval;
    m_nextFireDate = MonotonicTime::timePointFromNow(m_interval);
    m_runLoop->m_liveTimers.add(std::bit_cast<uintptr_t>(this));
    ::PostMessage(m_runLoop->m_runLoopMessageWindow, SetTimerMessage, std::bit_cast<uintptr_t>(this), interval.millisecondsAs<UINT>());
}

void RunLoop::TimerBase::stop()
{
    Locker locker { m_runLoop->m_loopLock };
    if (!isActiveWithLock())
        return;

    m_isActive = false;
    m_runLoop->m_liveTimers.remove(std::bit_cast<uintptr_t>(this));
    ::PostMessage(m_runLoop->m_runLoopMessageWindow, KillTimerMessage, std::bit_cast<uintptr_t>(this), 0LL);
}

bool RunLoop::TimerBase::isActiveWithLock() const
{
    return m_isActive;
}

bool RunLoop::TimerBase::isActive() const
{
    Locker locker { m_runLoop->m_loopLock };
    return isActiveWithLock();
}

Seconds RunLoop::TimerBase::secondsUntilFire() const
{
    Locker locker { m_runLoop->m_loopLock };
    if (isActiveWithLock())
        return std::max<Seconds>(m_nextFireDate - MonotonicTime::now(), 0_s);
    return 0_s;
}

} // namespace WTF
