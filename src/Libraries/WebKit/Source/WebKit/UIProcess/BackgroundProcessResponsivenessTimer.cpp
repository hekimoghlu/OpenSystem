/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 29, 2021.
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
#include "BackgroundProcessResponsivenessTimer.h"

#include "Logging.h"
#include "WebProcessMessages.h"
#include "WebProcessProxy.h"

namespace WebKit {

static const Seconds initialCheckingInterval { 20_s };
static const Seconds maximumCheckingInterval { 8_h };
static const Seconds responsivenessTimeout { 90_s };

BackgroundProcessResponsivenessTimer::BackgroundProcessResponsivenessTimer(WebProcessProxy& webProcessProxy)
    : m_webProcessProxy(webProcessProxy)
    , m_checkingInterval(initialCheckingInterval)
    , m_responsivenessCheckTimer(RunLoop::main(), this, &BackgroundProcessResponsivenessTimer::responsivenessCheckTimerFired)
    , m_timeoutTimer(RunLoop::main(), this, &BackgroundProcessResponsivenessTimer::timeoutTimerFired)
{
}

BackgroundProcessResponsivenessTimer::~BackgroundProcessResponsivenessTimer()
{
}

Ref<WebProcessProxy> BackgroundProcessResponsivenessTimer::protectedWebProcessProxy() const
{
    return const_cast<WebProcessProxy&>(m_webProcessProxy.get());
}

void BackgroundProcessResponsivenessTimer::updateState()
{
    if (!shouldBeActive()) {
        if (m_responsivenessCheckTimer.isActive()) {
            m_checkingInterval = initialCheckingInterval;
            m_responsivenessCheckTimer.stop();
        }
        m_timeoutTimer.stop();
        m_isResponsive = true;
        return;
    }

    if (!isActive())
        m_responsivenessCheckTimer.startOneShot(m_checkingInterval);
}

void BackgroundProcessResponsivenessTimer::didReceiveBackgroundResponsivenessPong()
{
    if (!m_timeoutTimer.isActive())
        return;

    m_timeoutTimer.stop();
    scheduleNextResponsivenessCheck();

    setResponsive(true);
}

void BackgroundProcessResponsivenessTimer::invalidate()
{
    m_timeoutTimer.stop();
    m_responsivenessCheckTimer.stop();
}

void BackgroundProcessResponsivenessTimer::processTerminated()
{
    invalidate();
    setResponsive(true);
}

void BackgroundProcessResponsivenessTimer::responsivenessCheckTimerFired()
{
    ASSERT(shouldBeActive());
    ASSERT(!m_timeoutTimer.isActive());

    m_timeoutTimer.startOneShot(responsivenessTimeout);
    protectedWebProcessProxy()->send(Messages::WebProcess::BackgroundResponsivenessPing(), 0);
}

void BackgroundProcessResponsivenessTimer::timeoutTimerFired()
{
    ASSERT(shouldBeActive());

    scheduleNextResponsivenessCheck();

    // This shouldn't happen but still check to be 100% sure we don't report
    // suspended processes as unresponsive.
    if (protectedWebProcessProxy()->throttler().isSuspended())
        return;

    if (!m_isResponsive)
        return;

    if (!client().mayBecomeUnresponsive())
        return;

    setResponsive(false);
}

void BackgroundProcessResponsivenessTimer::setResponsive(bool isResponsive)
{
    if (m_isResponsive == isResponsive)
        return;

    Ref protectedClient { client() };

    client().willChangeIsResponsive();
    m_isResponsive = isResponsive;
    client().didChangeIsResponsive();

    if (m_isResponsive) {
        RELEASE_LOG_ERROR(PerformanceLogging, "Notifying the client that background WebProcess with pid %d has become responsive again", m_webProcessProxy->processID());
        client().didBecomeResponsive();
    } else {
        RELEASE_LOG_ERROR(PerformanceLogging, "Notifying the client that background WebProcess with pid %d has become unresponsive", m_webProcessProxy->processID());
        client().didBecomeUnresponsive();
    }
}

bool BackgroundProcessResponsivenessTimer::shouldBeActive() const
{
#if !USE(RUNNINGBOARD)
    auto webProcess = protectedWebProcessProxy();
    if (webProcess->visiblePageCount())
        return false;
    if (webProcess->throttler().isSuspended())
        return false;
    if (webProcess->isStandaloneServiceWorkerProcess())
        return true;
    return webProcess->pageCount();
#else
    // Disable background process responsiveness checking when using RunningBoard since such processes usually get suspended.
    return false;
#endif
}

bool BackgroundProcessResponsivenessTimer::isActive() const
{
    return m_responsivenessCheckTimer.isActive() || m_timeoutTimer.isActive();
}

void BackgroundProcessResponsivenessTimer::scheduleNextResponsivenessCheck()
{
    // Exponential backoff to avoid waking up the process too often.
    ASSERT(!m_responsivenessCheckTimer.isActive());
    m_checkingInterval = std::min(m_checkingInterval * 2, maximumCheckingInterval);
    m_responsivenessCheckTimer.startOneShot(m_checkingInterval);
}

ResponsivenessTimer::Client& BackgroundProcessResponsivenessTimer::client() const
{
    return const_cast<WebProcessProxy&>(m_webProcessProxy.get());
}

} // namespace WebKit
