/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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
#ifndef ResponsivenessTimer_h
#define ResponsivenessTimer_h

#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#include <wtf/RefCounted.h>
#include <wtf/RunLoop.h>
#include <wtf/WeakRef.h>

namespace WebKit {

class ResponsivenessTimer : public RefCounted<ResponsivenessTimer> {
    WTF_MAKE_FAST_ALLOCATED;
public:
    class Client : public AbstractRefCountedAndCanMakeWeakPtr<Client> {
    public:
        virtual ~Client() { }
        virtual void didBecomeUnresponsive() = 0;
        virtual void didBecomeResponsive() = 0;

        virtual void willChangeIsResponsive() = 0;
        virtual void didChangeIsResponsive() = 0;

        virtual bool mayBecomeUnresponsive() = 0;
    };

    static constexpr Seconds defaultResponsivenessTimeout = 3_s;
    static Ref<ResponsivenessTimer> create(ResponsivenessTimer::Client&, Seconds responsivenessTimeout);
    ~ResponsivenessTimer();

    void start();

    // A responsiveness timer with lazy stop does not stop the underlying system timer when stopped.
    // Instead, it ignores the timeout if stop() was already called.
    //
    // This exists to reduce the rate at which we reset the timer.
    //
    // With a non lazy timer, we may set a timer and reset it soon after because the process is responsive.
    // For events, this means reseting a timer 120 times/s for a 60 Hz event source.
    // By not reseting the timer when responsive, we cut that in half to 60 timeout changes.
    void startWithLazyStop();

    void stop();

    void invalidate();

    // Return true if stop() was not called betfore the responsiveness timeout.
    bool isResponsive() const { return m_isResponsive; }

    // Return true if there is an active timer. The state could be responsive or not.
    bool hasActiveTimer() const { return m_waitingForTimer; }

    void processTerminated();

private:
    ResponsivenessTimer(ResponsivenessTimer::Client&, Seconds responsivenessTimeout);

    void timerFired();

    bool mayBecomeUnresponsive() const;

    WeakPtr<ResponsivenessTimer::Client> m_client;

    RunLoop::Timer m_timer;
    MonotonicTime m_restartFireTime;

    bool m_isResponsive { true };
    bool m_waitingForTimer { false };
    bool m_useLazyStop { false };
    Seconds m_responsivenessTimeout;
};

} // namespace WebKit

#endif // ResponsivenessTimer_h

