/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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
#import "config.h"
#import "RemoteScrollingUIState.h"

#import "ArgumentCoders.h"
#import "GeneratedSerializers.h"
#import <WebCore/ScrollingNodeID.h>
#import <wtf/text/TextStream.h>

namespace WebKit {

RemoteScrollingUIState::RemoteScrollingUIState(OptionSet<RemoteScrollingUIStateChanges> changes, HashSet<WebCore::ScrollingNodeID>&& nodesWithActiveScrollSnap, HashSet<WebCore::ScrollingNodeID>&& nodesWithActiveUserScrolls, HashSet<WebCore::ScrollingNodeID>&& nodesWithActiveRubberband)
    : m_changes(changes)
    , m_nodesWithActiveScrollSnap(WTFMove(nodesWithActiveScrollSnap))
    , m_nodesWithActiveUserScrolls(WTFMove(nodesWithActiveUserScrolls))
    , m_nodesWithActiveRubberband(WTFMove(nodesWithActiveRubberband))
{
}

void RemoteScrollingUIState::reset()
{
    clearChanges();
    m_nodesWithActiveScrollSnap.clear();
    m_nodesWithActiveUserScrolls.clear();
    m_nodesWithActiveRubberband.clear();
}

void RemoteScrollingUIState::addNodeWithActiveScrollSnap(WebCore::ScrollingNodeID nodeID)
{
    auto addResult = m_nodesWithActiveScrollSnap.add(nodeID);
    if (addResult.isNewEntry)
        m_changes.add(Changes::ScrollSnapNodes);
}

void RemoteScrollingUIState::removeNodeWithActiveScrollSnap(WebCore::ScrollingNodeID nodeID)
{
    if (m_nodesWithActiveScrollSnap.remove(nodeID))
        m_changes.add(Changes::ScrollSnapNodes);
}

void RemoteScrollingUIState::addNodeWithActiveUserScroll(WebCore::ScrollingNodeID nodeID)
{
    auto addResult = m_nodesWithActiveUserScrolls.add(nodeID);
    if (addResult.isNewEntry)
        m_changes.add(Changes::UserScrollNodes);
}

void RemoteScrollingUIState::removeNodeWithActiveUserScroll(WebCore::ScrollingNodeID nodeID)
{
    if (m_nodesWithActiveUserScrolls.remove(nodeID))
        m_changes.add(Changes::UserScrollNodes);
}

void RemoteScrollingUIState::clearNodesWithActiveUserScroll()
{
    if (m_nodesWithActiveUserScrolls.isEmpty())
        return;

    m_nodesWithActiveUserScrolls.clear();
    m_changes.add(Changes::UserScrollNodes);
}

void RemoteScrollingUIState::addNodeWithActiveRubberband(WebCore::ScrollingNodeID nodeID)
{
    auto addResult = m_nodesWithActiveRubberband.add(nodeID);
    if (addResult.isNewEntry)
        m_changes.add(Changes::RubberbandingNodes);
}

void RemoteScrollingUIState::removeNodeWithActiveRubberband(WebCore::ScrollingNodeID nodeID)
{
    if (m_nodesWithActiveRubberband.remove(nodeID))
        m_changes.add(Changes::RubberbandingNodes);
}

void RemoteScrollingUIState::clearNodesWithActiveRubberband()
{
    if (m_nodesWithActiveRubberband.isEmpty())
        return;

    m_nodesWithActiveRubberband.clear();
    m_changes.add(Changes::RubberbandingNodes);
}

} // namespace WebKit
