/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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
#include "ScrollingTreeGestureState.h"

#if ENABLE(ASYNC_SCROLLING)

#include "PlatformWheelEvent.h"
#include "ScrollingTree.h"

namespace WebCore {


ScrollingTreeGestureState::ScrollingTreeGestureState(ScrollingTree& scrollingTree)
    : m_scrollingTree(scrollingTree)
{
}

void ScrollingTreeGestureState::receivedWheelEvent(const PlatformWheelEvent& event)
{
    if (event.isGestureStart()) {
        clearAllNodes();
        return;
    }
}

bool ScrollingTreeGestureState::handleGestureCancel(const PlatformWheelEvent& event)
{
    if (event.isGestureCancel()) {
        if (m_mayBeginNodeID)
            m_scrollingTree.handleWheelEventPhase(*m_mayBeginNodeID, PlatformWheelEventPhase::Cancelled);
        return true;
    }
    
    return false;
}

void ScrollingTreeGestureState::nodeDidHandleEvent(ScrollingNodeID nodeID, const PlatformWheelEvent& event)
{
    switch (event.phase()) {
    case PlatformWheelEventPhase::MayBegin:
        m_mayBeginNodeID = nodeID;
        m_scrollingTree.handleWheelEventPhase(nodeID, event.phase());
        break;
    case PlatformWheelEventPhase::Cancelled:
        // We can get here for via handleWheelEventAfterMainThread(), in which case handleGestureCancel() was not called first.
        handleGestureCancel(event);
        break;
    case PlatformWheelEventPhase::Began:
        m_activeNodeID = nodeID;
        m_scrollingTree.handleWheelEventPhase(nodeID, event.phase());
        break;
    case PlatformWheelEventPhase::Ended:
        if (m_activeNodeID)
            m_scrollingTree.handleWheelEventPhase(*m_activeNodeID, event.phase());
        break;
    case PlatformWheelEventPhase::Changed:
    case PlatformWheelEventPhase::Stationary:
    case PlatformWheelEventPhase::None:
        break;
    }

    switch (event.momentumPhase()) {
    case PlatformWheelEventPhase::MayBegin:
    case PlatformWheelEventPhase::Cancelled:
        ASSERT_NOT_REACHED();
        break;
    case PlatformWheelEventPhase::Began:
        m_activeNodeID = nodeID;
        m_scrollingTree.handleWheelEventPhase(nodeID, event.momentumPhase());
        break;
    case PlatformWheelEventPhase::Ended:
        if (m_activeNodeID)
            m_scrollingTree.handleWheelEventPhase(*m_activeNodeID, event.momentumPhase());
        break;
    case PlatformWheelEventPhase::Changed:
    case PlatformWheelEventPhase::Stationary:
    case PlatformWheelEventPhase::None:
        break;
    }
}

void ScrollingTreeGestureState::clearAllNodes()
{
    m_mayBeginNodeID = std::nullopt;
    m_activeNodeID = std::nullopt;
}

};

#endif // ENABLE(ASYNC_SCROLLING)
