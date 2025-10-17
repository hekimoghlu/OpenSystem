/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 31, 2021.
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

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

// This class is used to detect when we are painting frequently so that - even in a painting model
// without display lists - we can build and cache portions of display lists and reuse them only when
// animating. Once we transition fully to display lists, we can probably just pull from the previous
// paint's display list if it is still around and get rid of this code.
class PaintFrequencyTracker {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(PaintFrequencyTracker);

public:
    PaintFrequencyTracker() = default;

    void track(MonotonicTime timestamp)
    {
        static unsigned paintFrequencyPaintCountThreshold = 20;
        static Seconds paintFrequencySecondsIdleThreshold = 5_s;

        if (!timestamp)
            timestamp = MonotonicTime::now();

        // Start by assuming the paint frequency is low
        m_paintFrequency = PaintFrequency::Low;

        if (timestamp - m_lastPaintTime > paintFrequencySecondsIdleThreshold) {
            // It has been 5 seconds since last time we draw this renderer. Reset the state
            // of this object as if, we've just started tracking the paint frequency.
            m_totalPaints = 0;
        } else if (m_totalPaints >= paintFrequencyPaintCountThreshold) {
            // Change the paint frequency to be high if this renderer has been painted at least 20 times.
            m_paintFrequency = PaintFrequency::High;
        }

        m_lastPaintTime = timestamp;
        ++m_totalPaints;
    }

    bool paintingFrequently() const { return m_paintFrequency == PaintFrequency::High; }

private:
    MonotonicTime m_lastPaintTime;
    unsigned m_totalPaints { 0 };

    enum class PaintFrequency : bool { Low, High };
    PaintFrequency m_paintFrequency { PaintFrequency::Low };
};

} // namespace WebCore
