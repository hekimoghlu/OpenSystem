/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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

#if ENABLE(CONTENT_CHANGE_OBSERVER)

#include "Timer.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashSet.h>

namespace WebCore {

class DOMTimer;

class DOMTimerHoldingTank final : public CanMakeCheckedPtr<DOMTimerHoldingTank> {
    WTF_MAKE_TZONE_ALLOCATED(DOMTimerHoldingTank);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DOMTimerHoldingTank);
public:
    DOMTimerHoldingTank();
    ~DOMTimerHoldingTank();

    void add(const DOMTimer&);
    void remove(const DOMTimer&);
    bool contains(const DOMTimer&);
    WEBCORE_EXPORT void removeAll();

private:
    void stopExceededMaximumHoldTimer();

    WeakHashSet<DOMTimer> m_timers;
    Timer m_exceededMaximumHoldTimer;
};

class DeferDOMTimersForScope {
public:
    explicit DeferDOMTimersForScope(bool enable)
        : m_previousIsDeferring { s_isDeferring }
    {
        if (enable)
            s_isDeferring = true;
    }

    ~DeferDOMTimersForScope() { s_isDeferring = m_previousIsDeferring; }

    static bool isDeferring() { return s_isDeferring; }

private:
    bool m_previousIsDeferring;
    static bool s_isDeferring;
};

} // namespace WebCore

#endif // ENABLE(CONTENT_CHANGE_OBSERVER)
