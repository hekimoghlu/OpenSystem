/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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

#include <WebCore/ProcessQualified.h>
#include <wtf/HashSet.h>
#include <wtf/OptionSet.h>
#include <wtf/TZoneMalloc.h>

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebCore {
struct ScrollingNodeIDType;
using ScrollingNodeID = ProcessQualified<ObjectIdentifier<ScrollingNodeIDType>>;
}

namespace WebKit {

enum class RemoteScrollingUIStateChanges : uint8_t {
    ScrollSnapNodes     = 1 << 0,
    UserScrollNodes     = 1 << 1,
    RubberbandingNodes  = 1 << 2,
};

class RemoteScrollingUIState {
    WTF_MAKE_TZONE_ALLOCATED(RemoteScrollingUIState);
public:
    using Changes = RemoteScrollingUIStateChanges;

    RemoteScrollingUIState() = default;
    RemoteScrollingUIState(OptionSet<RemoteScrollingUIStateChanges>, HashSet<WebCore::ScrollingNodeID>&& nodesWithActiveScrollSnap, HashSet<WebCore::ScrollingNodeID>&& nodesWithActiveUserScrolls, HashSet<WebCore::ScrollingNodeID>&& nodesWithActiveRubberband);

    OptionSet<RemoteScrollingUIStateChanges> changes() const { return m_changes; }
    void clearChanges() { m_changes = { }; }
    
    void reset();

    const HashSet<WebCore::ScrollingNodeID>& nodesWithActiveScrollSnap() const { return m_nodesWithActiveScrollSnap; }
    void addNodeWithActiveScrollSnap(WebCore::ScrollingNodeID);
    void removeNodeWithActiveScrollSnap(WebCore::ScrollingNodeID);
    
    const HashSet<WebCore::ScrollingNodeID>& nodesWithActiveUserScrolls() const { return m_nodesWithActiveUserScrolls; }
    void addNodeWithActiveUserScroll(WebCore::ScrollingNodeID);
    void removeNodeWithActiveUserScroll(WebCore::ScrollingNodeID);
    void clearNodesWithActiveUserScroll();

    const HashSet<WebCore::ScrollingNodeID>& nodesWithActiveRubberband() const { return m_nodesWithActiveRubberband; }
    void addNodeWithActiveRubberband(WebCore::ScrollingNodeID);
    void removeNodeWithActiveRubberband(WebCore::ScrollingNodeID);
    void clearNodesWithActiveRubberband();

private:
    OptionSet<RemoteScrollingUIStateChanges> m_changes;
    HashSet<WebCore::ScrollingNodeID> m_nodesWithActiveScrollSnap;
    HashSet<WebCore::ScrollingNodeID> m_nodesWithActiveUserScrolls;
    HashSet<WebCore::ScrollingNodeID> m_nodesWithActiveRubberband;
};

} // namespace WebKit
