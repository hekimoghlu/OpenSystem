/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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
#include "MainThreadSharedTimer.h"

#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

#if USE(GLIB)
#include <wtf/glib/RunLoopSourcePriority.h>
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MainThreadSharedTimer);

MainThreadSharedTimer& MainThreadSharedTimer::singleton()
{
    static NeverDestroyed<MainThreadSharedTimer> instance;
    return instance;
}

#if USE(CF) || OS(WINDOWS)
MainThreadSharedTimer::MainThreadSharedTimer() = default;
#else
MainThreadSharedTimer::MainThreadSharedTimer()
    : m_timer(RunLoop::main(), this, &MainThreadSharedTimer::fired)
{
#if USE(GLIB)
    m_timer.setPriority(RunLoopSourcePriority::MainThreadSharedTimer);
    m_timer.setName("[WebKit] MainThreadSharedTimer"_s);
#endif
}

void MainThreadSharedTimer::setFireInterval(Seconds interval)
{
    ASSERT(m_firedFunction);
    m_timer.startOneShot(interval);
}

void MainThreadSharedTimer::stop()
{
    m_timer.stop();
}

void MainThreadSharedTimer::invalidate()
{
}
#endif

void MainThreadSharedTimer::setFiredFunction(Function<void()>&& firedFunction)
{
    RELEASE_ASSERT(!m_firedFunction || !firedFunction);
    m_firedFunction = WTFMove(firedFunction);
}

void MainThreadSharedTimer::fired()
{
    ASSERT(m_firedFunction);
    m_firedFunction();
}

} // namespace WebCore
