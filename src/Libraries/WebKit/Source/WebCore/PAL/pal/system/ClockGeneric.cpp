/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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
#include "ClockGeneric.h"

#include <wtf/TZoneMallocInlines.h>

namespace PAL {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ClockGeneric);

std::unique_ptr<Clock> Clock::create()
{
    return makeUnique<ClockGeneric>();
}

ClockGeneric::ClockGeneric()
    : m_running(false)
    , m_rate(1)
    , m_offset(0)
{
    m_startTime = m_lastTime = now();
}

void ClockGeneric::setCurrentTime(double time)
{
    m_startTime = m_lastTime = now();
    m_offset = time;
}

double ClockGeneric::currentTime() const
{
    return currentDelta() + m_offset;
}

void ClockGeneric::setPlayRate(double rate)
{
    m_offset += currentDelta();
    m_lastTime = m_startTime = now();
    m_rate = rate;
}

void ClockGeneric::start()
{
    if (m_running)
        return;

    m_lastTime = m_startTime = now();
    m_running = true;
}

void ClockGeneric::stop()
{
    if (!m_running)
        return;

    m_offset += currentDelta();
    m_lastTime = m_startTime = now();
    m_running = false;
}

MonotonicTime ClockGeneric::now() const
{
    return MonotonicTime::now();
}

double ClockGeneric::currentDelta() const
{
    if (m_running)
        m_lastTime = now();
    return (m_lastTime - m_startTime).seconds() * m_rate;
}

}
