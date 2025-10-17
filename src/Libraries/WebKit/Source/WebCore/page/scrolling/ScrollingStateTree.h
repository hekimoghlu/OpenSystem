/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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

#include "ScrollingStateNode.h"
#include <wtf/CheckedPtr.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
 
namespace WebCore {

class AsyncScrollingCoordinator;
class ScrollingStateFrameScrollingNode;

// The ScrollingStateTree is a tree that manages ScrollingStateNodes. The nodes keep track of the current
// state of scrolling related properties. Whenever any properties change, the scrolling coordinator
// will be informed and will schedule a timer that will clone the new state tree and send it over to
// the scrolling thread, avoiding locking. 

class ScrollingStateTree final : public CanMakeCheckedPtr<ScrollingStateTree> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ScrollingStateTree, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ScrollingStateTree);
    friend class ScrollingStateNode;
public:
    WEBCORE_EXPORT static std::optional<ScrollingStateTree> createAfterReconstruction(bool, bool, RefPtr<ScrollingStateFrameScrollingNode>&&);
    WEBCORE_EXPORT ScrollingStateTree(AsyncScrollingCoordinator* = nullptr);
    WEBCORE_EXPORT ScrollingStateTree(ScrollingStateTree&&);
    WEBCORE_EXPORT ~ScrollingStateTree();

    WEBCORE_EXPORT RefPtr<ScrollingStateFrameScrollingNode> rootStateNode() const;
    WEBCORE_EXPORT RefPtr<ScrollingStateNode> stateNodeForID(std::optional<ScrollingNodeID>) const;

    ScrollingNodeID createUnparentedNode(ScrollingNodeType, ScrollingNodeID);
    WEBCORE_EXPORT std::optional<ScrollingNodeID> insertNode(ScrollingNodeType, ScrollingNodeID, std::optional<ScrollingNodeID> parentID, size_t childIndex);
    void unparentNode(std::optional<ScrollingNodeID>);
    void unparentChildrenAndDestroyNode(std::optional<ScrollingNodeID>);
    void detachAndDestroySubtree(std::optional<ScrollingNodeID>);
    void clear();

    // Copies the current tree state and clears the changed properties mask in the original.
    WEBCORE_EXPORT std::unique_ptr<ScrollingStateTree> commit(LayerRepresentation::Type preferredLayerRepresentation);

    WEBCORE_EXPORT void attachDeserializedNodes();

    WEBCORE_EXPORT void setHasChangedProperties(bool = true);
    bool hasChangedProperties() const { return m_hasChangedProperties; }

    bool hasNewRootStateNode() const { return m_hasNewRootStateNode; }

    unsigned nodeCount() const { return m_stateNodeMap.size(); }
    unsigned scrollingNodeCount() const { return m_scrollingNodeCount; }

    using StateNodeMap = UncheckedKeyHashMap<ScrollingNodeID, Ref<ScrollingStateNode>>;
    const StateNodeMap& nodeMap() const { return m_stateNodeMap; }

    LayerRepresentation::Type preferredLayerRepresentation() const { return m_preferredLayerRepresentation; }
    void setPreferredLayerRepresentation(LayerRepresentation::Type representation) { m_preferredLayerRepresentation = representation; }

    void reconcileViewportConstrainedLayerPositions(std::optional<ScrollingNodeID>, const LayoutRect& viewportRect, ScrollingLayerPositionAction);

    void scrollingNodeAdded()
    {
        ++m_scrollingNodeCount;
    }
    void scrollingNodeRemoved()
    {
        ASSERT(m_scrollingNodeCount);
        --m_scrollingNodeCount;
    }

    WEBCORE_EXPORT String scrollingStateTreeAsText(OptionSet<ScrollingStateTreeAsTextBehavior>) const;
    FrameIdentifier rootFrameIdentifier() const { return *m_rootFrameIdentifier; }
    void setRootFrameIdentifier(std::optional<FrameIdentifier> frameID) { m_rootFrameIdentifier = frameID; }

private:
    ScrollingStateTree(bool hasNewRootStateNode, bool hasChangedProperties, RefPtr<ScrollingStateFrameScrollingNode>&&);

    void setRootStateNode(Ref<ScrollingStateFrameScrollingNode>&&);
    void addNode(ScrollingStateNode&);

    Ref<ScrollingStateNode> createNode(ScrollingNodeType, ScrollingNodeID);

    void removeNodeAndAllDescendants(ScrollingStateNode&);

    void recursiveNodeWillBeRemoved(ScrollingStateNode&);
    void willRemoveNode(ScrollingStateNode&);
    
    bool isValid() const;
    void traverse(const ScrollingStateNode&, const Function<void(const ScrollingStateNode&)>&) const;

    ThreadSafeWeakPtr<AsyncScrollingCoordinator> m_scrollingCoordinator;
    Markable<FrameIdentifier> m_rootFrameIdentifier;

    // Contains all the nodes we know about (those in the m_rootStateNode tree, and in m_unparentedNodes subtrees).
    StateNodeMap m_stateNodeMap;
    // Owns roots of unparented subtrees.
    UncheckedKeyHashMap<ScrollingNodeID, RefPtr<ScrollingStateNode>> m_unparentedNodes;

    RefPtr<ScrollingStateFrameScrollingNode> m_rootStateNode;
    unsigned m_scrollingNodeCount { 0 };
    LayerRepresentation::Type m_preferredLayerRepresentation { LayerRepresentation::GraphicsLayerRepresentation };
    bool m_hasChangedProperties { false };
    bool m_hasNewRootStateNode { false };
};

} // namespace WebCore

#ifndef NDEBUG
void showScrollingStateTree(const WebCore::ScrollingStateTree&);
void showScrollingStateTree(const WebCore::ScrollingStateNode&);
#endif

#endif // ENABLE(ASYNC_SCROLLING)
