/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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
#ifndef TimerEventBasedMock_h
#define TimerEventBasedMock_h

#if ENABLE(WEB_RTC)

#include "Timer.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

class TimerEvent;

class MockNotifier : public RefCounted<MockNotifier> {
public:
    virtual ~MockNotifier() = default;
    virtual void fire() = 0;
};

class TimerEventBasedMock {
public:
    void removeEvent(TimerEvent& event)
    {
        size_t pos = m_timerEvents.find(&event);
        m_timerEvents.remove(pos);
    }

protected:
    Vector<RefPtr<TimerEvent> > m_timerEvents;
};

class TimerEvent : public RefCounted<TimerEvent> {
public:
    TimerEvent(TimerEventBasedMock* mock, Ref<MockNotifier>&& notifier)
        : m_mock(mock)
        , m_timer(*this, &TimerEvent::timerFired)
        , m_notifier(WTFMove(notifier))
    {
        m_timer.startOneShot(500_ms);
    }

    virtual ~TimerEvent()
    {
        m_mock = nullptr;
    }

    void timerFired()
    {
        m_notifier->fire();
        m_mock->removeEvent(*this);
    }

private:
    TimerEventBasedMock* m_mock;
    Timer m_timer;
    Ref<MockNotifier> m_notifier;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)

#endif // TimerEventBasedMock_h
