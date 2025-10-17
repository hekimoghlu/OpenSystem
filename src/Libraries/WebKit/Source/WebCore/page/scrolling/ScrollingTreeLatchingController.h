/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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

#if ENABLE(ASYNC_SCROLLING)

#include "ScrollTypes.h"
#include <wtf/Lock.h>
#include <wtf/Markable.h>
#include <wtf/MonotonicTime.h>
#include <wtf/OptionSet.h>

namespace WebCore {

class PlatformWheelEvent;
enum class WheelEventProcessingSteps : uint8_t;

class ScrollingTreeLatchingController {
public:
    struct ScrollingNodeAndProcessingSteps {
        ScrollingNodeID scrollingNodeID;
        OptionSet<WheelEventProcessingSteps> processingSteps;
    };

    ScrollingTreeLatchingController();

    void receivedWheelEvent(const PlatformWheelEvent&, OptionSet<WheelEventProcessingSteps>, bool allowLatching);

    std::optional<ScrollingNodeAndProcessingSteps> latchingDataForEvent(const PlatformWheelEvent&, bool allowLatching) const;
    void nodeDidHandleEvent(ScrollingNodeID, OptionSet<WheelEventProcessingSteps>, const PlatformWheelEvent&, bool allowLatching);

    std::optional<ScrollingNodeID> latchedNodeID() const;
    std::optional<ScrollingNodeAndProcessingSteps> latchedNodeAndSteps() const;

    void nodeWasRemoved(ScrollingNodeID);
    void clearLatchedNode();

private:
    bool latchedNodeIsRelevant() const;

    mutable Lock m_latchedNodeLock;
    std::optional<ScrollingNodeAndProcessingSteps> m_latchedNodeAndSteps WTF_GUARDED_BY_LOCK(m_latchedNodeLock);
    std::optional<OptionSet<WheelEventProcessingSteps>> m_processingStepsForCurrentGesture;
    MonotonicTime m_lastLatchedNodeInterationTime;
};

}

#endif // ENABLE(ASYNC_SCROLLING)
