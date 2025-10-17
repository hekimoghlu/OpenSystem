/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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
#include "ResponsivenessTimer.h"

namespace WebKit {

Ref<ResponsivenessTimer> ResponsivenessTimer::create(ResponsivenessTimer::Client& client, Seconds responsivenessTimeout)
{
    return adoptRef(*new ResponsivenessTimer(client, responsivenessTimeout));
}

ResponsivenessTimer::ResponsivenessTimer(ResponsivenessTimer::Client& client, Seconds responsivenessTimeout)
    : m_client(client)
    , m_timer(RunLoop::main(), this, &ResponsivenessTimer::timerFired)
    , m_responsivenessTimeout(responsivenessTimeout)
{
}

ResponsivenessTimer::~ResponsivenessTimer() = default;

void ResponsivenessTimer::invalidate()
{
    m_timer.stop();
    m_restartFireTime = MonotonicTime();
    m_waitingForTimer = false;
    m_useLazyStop = false;
}

void ResponsivenessTimer::timerFired()
{
    if (!m_waitingForTimer)
        return;

    RefPtr client = m_client.get();
    if (!client)
        return;

    if (m_restartFireTime) {
        MonotonicTime now = MonotonicTime::now();
        MonotonicTime restartFireTime = m_restartFireTime;
        m_restartFireTime = MonotonicTime();

        if (restartFireTime > now) {
            m_timer.startOneShot(restartFireTime - now);
            return;
        }
    }

    m_waitingForTimer = false;
    m_useLazyStop = false;

    if (!m_isResponsive)
        return;

    if (!mayBecomeUnresponsive()) {
        m_waitingForTimer = true;
        m_timer.startOneShot(m_responsivenessTimeout);
        return;
    }

    client->willChangeIsResponsive();
    m_isResponsive = false;
    client->didChangeIsResponsive();

    client->didBecomeUnresponsive();
}
    
void ResponsivenessTimer::start()
{
    if (m_waitingForTimer)
        return;

    m_waitingForTimer = true;
    m_useLazyStop = false;

    if (m_timer.isActive()) {
        // The timer is still active from a lazy stop.
        // Instead of restarting the timer, we schedule a new delay after this one finishes.
        //
        // In most cases, stop is called before we get to schedule the second timer, saving us
        // the scheduling of the timer entirely.
        m_restartFireTime = MonotonicTime::now() + m_responsivenessTimeout;
    } else {
        m_restartFireTime = MonotonicTime();
        m_timer.startOneShot(m_responsivenessTimeout);
    }
}

bool ResponsivenessTimer::mayBecomeUnresponsive() const
{
#if !defined(NDEBUG) || ASAN_ENABLED
    return false;
#else
    static bool isLibgmallocEnabled = [] {
        char* variable = getenv("DYLD_INSERT_LIBRARIES");
        if (!variable)
            return false;
        if (!contains(unsafeSpan(variable), "libgmalloc"_span))
            return false;
        return true;
    }();
    if (isLibgmallocEnabled)
        return false;

    RefPtr client = m_client.get();
    return client && client->mayBecomeUnresponsive();
#endif
}

void ResponsivenessTimer::startWithLazyStop()
{
    if (!m_waitingForTimer) {
        start();
        m_useLazyStop = true;
    }
}

void ResponsivenessTimer::stop()
{
    if (!m_isResponsive) {
        if (RefPtr client = m_client.get()) {
            // We got a life sign from the web process.
            client->willChangeIsResponsive();
            m_isResponsive = true;
            client->didChangeIsResponsive();

            client->didBecomeResponsive();
        }
    }

    m_waitingForTimer = false;

    if (m_useLazyStop)
        m_useLazyStop = false;
    else
        m_timer.stop();
}

void ResponsivenessTimer::processTerminated()
{
    invalidate();
}

} // namespace WebKit
