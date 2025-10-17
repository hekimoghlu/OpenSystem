/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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

#if PLATFORM(IOS_FAMILY)

#include "ScrollAnimator.h"

#include "IntPoint.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class PlatformTouchEvent;

class ScrollAnimatorIOS : public ScrollAnimator {
    WTF_MAKE_TZONE_ALLOCATED(ScrollAnimatorIOS);
public:
    ScrollAnimatorIOS(ScrollableArea&);
    virtual ~ScrollAnimatorIOS();

#if ENABLE(TOUCH_EVENTS)
    bool handleTouchEvent(const PlatformTouchEvent&) override;
#endif

private:
#if ENABLE(TOUCH_EVENTS)
    void determineScrollableAreaForTouchSequence(const IntSize& touchDelta);

    // State for handling sequences of touches in defaultTouchEventHandler.
    enum AxisLatch {
        AxisLatchNotComputed,
        AxisLatchNone,
        AxisLatchHorizontal,
        AxisLatchVertical
    };
    AxisLatch m_touchScrollAxisLatch { AxisLatchNotComputed };
    bool m_inTouchSequence { false };
    bool m_committedToScrollAxis { false };
    bool m_startedScroll { false };
    IntPoint m_firstTouchPoint;
    IntPoint m_lastTouchPoint;

    // When we're in a touch sequence, this will point to the scrollable area that
    // should actually be scrolled during the sequence.
    ScrollableArea* m_scrollableAreaForTouchSequence { nullptr };
#endif
};

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
