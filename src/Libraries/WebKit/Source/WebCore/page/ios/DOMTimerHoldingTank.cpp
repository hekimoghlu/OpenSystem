/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
#include "DOMTimerHoldingTank.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(CONTENT_CHANGE_OBSERVER)

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DOMTimerHoldingTank);

#if PLATFORM(IOS_SIMULATOR)
constexpr Seconds maximumHoldTimeLimit { 50_ms };
#else
constexpr Seconds maximumHoldTimeLimit { 32_ms };
#endif

bool DeferDOMTimersForScope::s_isDeferring { false };

DOMTimerHoldingTank::DOMTimerHoldingTank()
    : m_exceededMaximumHoldTimer { *this, &DOMTimerHoldingTank::removeAll }
{
}

DOMTimerHoldingTank::~DOMTimerHoldingTank() = default;

void DOMTimerHoldingTank::add(const DOMTimer& timer)
{
    m_timers.add(timer);
    if (!m_exceededMaximumHoldTimer.isActive())
        m_exceededMaximumHoldTimer.startOneShot(maximumHoldTimeLimit);
}

void DOMTimerHoldingTank::remove(const DOMTimer& timer)
{
    stopExceededMaximumHoldTimer();
    m_timers.remove(timer);
}

bool DOMTimerHoldingTank::contains(const DOMTimer& timer)
{
    return m_timers.contains(timer);
}

void DOMTimerHoldingTank::removeAll()
{
    stopExceededMaximumHoldTimer();
    m_timers.clear();
}

inline void DOMTimerHoldingTank::stopExceededMaximumHoldTimer()
{
    if (m_exceededMaximumHoldTimer.isActive())
        m_exceededMaximumHoldTimer.stop();
}

} // namespace WebCore

#endif // ENABLE(CONTENT_CHANGE_OBSERVER)
