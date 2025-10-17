/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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
#include "DisplayVBlankMonitor.h"

#include "DisplayVBlankMonitorTimer.h"
#include "Logging.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Threading.h>
#include <wtf/Vector.h>
#include <wtf/glib/RunLoopSourcePriority.h>

#if USE(LIBDRM)
#include "DisplayVBlankMonitorDRM.h"
#endif

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DisplayVBlankMonitor);

std::unique_ptr<DisplayVBlankMonitor> DisplayVBlankMonitor::create(PlatformDisplayID displayID)
{
    static const char* forceTimer = getenv("WEBKIT_FORCE_VBLANK_TIMER");
    if (!displayID || (forceTimer && strcmp(forceTimer, "0")))
        return DisplayVBlankMonitorTimer::create();

#if USE(LIBDRM)
    if (auto monitor = DisplayVBlankMonitorDRM::create(displayID))
        return monitor;
    RELEASE_LOG_FAULT(DisplayLink, "Failed to create DRM vblank monitor, falling back to timer");
#else
    UNUSED_PARAM(displayID);
#endif
    return DisplayVBlankMonitorTimer::create();
}

DisplayVBlankMonitor::DisplayVBlankMonitor(unsigned refreshRate)
    : m_refreshRate(refreshRate)
    , m_destroyThreadTimer(RunLoop::main(), this, &DisplayVBlankMonitor::destroyThreadTimerFired)
{
    m_destroyThreadTimer.setPriority(RunLoopSourcePriority::ReleaseUnusedResourcesTimer);
}

DisplayVBlankMonitor::~DisplayVBlankMonitor()
{
    ASSERT(!m_thread);
}

bool DisplayVBlankMonitor::startThreadIfNeeded()
{
    if (m_thread)
        return false;

    m_thread = Thread::create("VBlankMonitor"_s, [this] {
        while (true) {
            {
                Locker locker { m_lock };
                m_condition.wait(m_lock, [this]() -> bool {
                    return m_state != State::Stop;
                });
                if (m_state == State::Invalid || m_state == State::Failed)
                    return;
            }

            if (!waitForVBlank()) {
                WTFLogAlways("Failed to wait for vblank");
                Locker locker { m_lock };
                m_state = State::Failed;
                return;
            }

            bool active;
            {
                Locker locker { m_lock };
                active = m_state == State::Active;
            }
            if (active)
                m_handler();
        }
    }, ThreadType::Graphics, Thread::QOS::Default);
    return true;
}

void DisplayVBlankMonitor::start()
{
    Locker locker { m_lock };
    if (m_state == State::Active)
        return;

    ASSERT(m_handler);
    m_state = State::Active;
    m_destroyThreadTimer.stop();
    if (!startThreadIfNeeded())
        m_condition.notifyAll();
}

void DisplayVBlankMonitor::stop()
{
    Locker locker { m_lock };
    if (m_state != State::Active)
        return;

    m_state = State::Stop;
    if (m_thread)
        m_destroyThreadTimer.startOneShot(30_s);
}

void DisplayVBlankMonitor::invalidate()
{
    if (!m_thread) {
        m_state = State::Invalid;
        return;
    }

    {
        Locker locker { m_lock };
        m_state = State::Invalid;
        m_condition.notifyAll();
    }
    m_thread->waitForCompletion();
    m_thread = nullptr;
}

bool DisplayVBlankMonitor::isActive()
{
    Locker locker { m_lock };
    return m_state == State::Active;
}

void DisplayVBlankMonitor::setHandler(Function<void()>&& handler)
{
    Locker locker { m_lock };
    ASSERT(m_state != State::Active);
    m_handler = WTFMove(handler);
}

void DisplayVBlankMonitor::destroyThreadTimerFired()
{
    if (!m_thread)
        return;

    invalidate();
}

} // namespace WebKit
