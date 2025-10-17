/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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
#include "ScrollingTreeLatchingController.h"

#if ENABLE(ASYNC_SCROLLING)

#include "FloatPoint.h"
#include "Logging.h"
#include "PlatformWheelEvent.h"
#include "ScrollingThread.h"
#include "ScrollingTree.h"

namespace WebCore {

// See also ScrollLatchingController.cpp
static const Seconds resetLatchedStateTimeout { 100_ms };

ScrollingTreeLatchingController::ScrollingTreeLatchingController() = default;

void ScrollingTreeLatchingController::receivedWheelEvent(const PlatformWheelEvent& wheelEvent, OptionSet<WheelEventProcessingSteps> processingSteps, bool allowLatching)
{
    if (!allowLatching)
        return;

    Locker locker { m_latchedNodeLock };
    if (wheelEvent.isGestureStart() && !latchedNodeIsRelevant()) {
        if (m_latchedNodeAndSteps) {
            LOG_WITH_STREAM(ScrollLatching, stream << "ScrollingTreeLatchingController " << this << " receivedWheelEvent - " << (MonotonicTime::now() - m_lastLatchedNodeInterationTime).milliseconds() << "ms since last event, clearing latched node");
            m_latchedNodeAndSteps.reset();
        }
        m_processingStepsForCurrentGesture = processingSteps;
    }
}

std::optional<ScrollingTreeLatchingController::ScrollingNodeAndProcessingSteps> ScrollingTreeLatchingController::latchingDataForEvent(const PlatformWheelEvent& wheelEvent, bool allowLatching) const
{
    if (!allowLatching)
        return std::nullopt;

    Locker locker { m_latchedNodeLock };

    // If we have a latched node, use it.
    if (wheelEvent.useLatchedEventElement() && m_latchedNodeAndSteps && latchedNodeIsRelevant()) {
        LOG_WITH_STREAM(ScrollLatching, stream << "ScrollingTreeLatchingController " << this << " latchedNodeForEvent: returning " << m_latchedNodeAndSteps->scrollingNodeID);
        return m_latchedNodeAndSteps;
    }

    return std::nullopt;
}

std::optional<ScrollingNodeID> ScrollingTreeLatchingController::latchedNodeID() const
{
    Locker locker { m_latchedNodeLock };
    if (m_latchedNodeAndSteps)
        return m_latchedNodeAndSteps->scrollingNodeID;

    return std::nullopt;
}

std::optional<ScrollingTreeLatchingController::ScrollingNodeAndProcessingSteps> ScrollingTreeLatchingController::latchedNodeAndSteps() const
{
    Locker locker { m_latchedNodeLock };
    return m_latchedNodeAndSteps;
}

void ScrollingTreeLatchingController::nodeDidHandleEvent(ScrollingNodeID scrollingNodeID, OptionSet<WheelEventProcessingSteps> processingSteps, const PlatformWheelEvent& wheelEvent, bool allowLatching)
{
    if (!allowLatching)
        return;

    Locker locker { m_latchedNodeLock };

    if (wheelEvent.useLatchedEventElement() && m_latchedNodeAndSteps && m_latchedNodeAndSteps->scrollingNodeID == scrollingNodeID) {
        if (wheelEvent.isEndOfMomentumScroll())
            m_lastLatchedNodeInterationTime = { };
        else
            m_lastLatchedNodeInterationTime = MonotonicTime::now();
        return;
    }

    auto shouldLatch = [&]() {
        if (wheelEvent.delta().isZero())
            return false;

        if (wheelEvent.isGestureStart())
            return true;

        if (!wheelEvent.isGestureContinuation())
            return false;

        if (valueOrDefault(m_processingStepsForCurrentGesture).contains(WheelEventProcessingSteps::SynchronousScrolling) && processingSteps.contains(WheelEventProcessingSteps::AsyncScrolling))
            return true;

        return false;
    };
    
    if (!shouldLatch())
        return;

    m_processingStepsForCurrentGesture = processingSteps;

    LOG_WITH_STREAM(ScrollLatching, stream << "ScrollingTreeLatchingController " << this << " nodeDidHandleEvent: latching to " << scrollingNodeID);
    m_latchedNodeAndSteps = ScrollingNodeAndProcessingSteps { scrollingNodeID, processingSteps };
    m_lastLatchedNodeInterationTime = MonotonicTime::now();
}

void ScrollingTreeLatchingController::nodeWasRemoved(ScrollingNodeID nodeID)
{
    Locker locker { m_latchedNodeLock };
    if (m_latchedNodeAndSteps && m_latchedNodeAndSteps->scrollingNodeID == nodeID)
        m_latchedNodeAndSteps.reset();
}

void ScrollingTreeLatchingController::clearLatchedNode()
{
    Locker locker { m_latchedNodeLock };
    LOG_WITH_STREAM(ScrollLatching, stream << "ScrollingTreeLatchingController " << this << " clearLatchedNode");
    m_latchedNodeAndSteps.reset();
}

bool ScrollingTreeLatchingController::latchedNodeIsRelevant() const
{
    auto secondsSinceLastInteraction = MonotonicTime::now() - m_lastLatchedNodeInterationTime;
    return secondsSinceLastInteraction < resetLatchedStateTimeout;
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
