/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 10, 2022.
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

#include "FloatSize.h"
#include "ScrollTypes.h"
#include "Timer.h"
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

#if ENABLE(WHEEL_EVENT_LATCHING)

namespace WTF {
class TextStream;
}

namespace WebCore {

class Element;
class LocalFrame;
class PlatformWheelEvent;
class ScrollableArea;
class WeakPtrImplWithEventTargetData;

class ScrollLatchingController {
    WTF_MAKE_TZONE_ALLOCATED(ScrollLatchingController);
public:
    explicit ScrollLatchingController(Page&);
    ~ScrollLatchingController();

    void clear();

    void ref() const;
    void deref() const;

    void receivedWheelEvent(const PlatformWheelEvent&);
    FloatSize cumulativeEventDelta() const { return m_cumulativeEventDelta; }

    // Frame containing latched scroller (may be the frame or some sub-scroller).
    LocalFrame* latchedFrame() const;

    // Returns true if no frame is latched, or latching is in the given frame (in which case latchedScroller will be non-null).
    bool latchingAllowsScrollingInFrame(const LocalFrame&, WeakPtr<ScrollableArea>& latchedScroller) const;

    void updateAndFetchLatchingStateForFrame(LocalFrame&, const PlatformWheelEvent&, RefPtr<Element>& latchedElement, WeakPtr<ScrollableArea>&, bool& isOverWidget);

    void removeLatchingStateForTarget(const Element&);
    void removeLatchingStateForFrame(const LocalFrame&);

    void dump(WTF::TextStream&) const;

private:
    struct FrameState {
        WeakPtr<Element, WeakPtrImplWithEventTargetData> wheelEventElement;
        WeakPtr<ScrollableArea> scrollableArea;
        LocalFrame* frame { nullptr };
        bool isOverWidget { false };
    };

    void clearOrScheduleClearIfNeeded(const PlatformWheelEvent&);
    void clearTimerFired();

    bool hasStateForFrame(const LocalFrame&) const;
    FrameState* stateForFrame(const LocalFrame&);
    const FrameState* stateForFrame(const LocalFrame&) const;

    bool shouldLatchToScrollableArea(const LocalFrame&, ScrollableArea*, FloatSize) const;

    WeakRef<Page> m_page;
    FloatSize m_cumulativeEventDelta;
    Vector<FrameState> m_frameStateStack;
    Timer m_clearLatchingStateTimer;
};

WTF::TextStream& operator<<(WTF::TextStream&, const ScrollLatchingController&);

} // namespace WebCore

#endif // ENABLE(WHEEL_EVENT_LATCHING)

