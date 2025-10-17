/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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

#include "IntRect.h"
#include "ScrollTypes.h"
#include "ScrollingCoordinator.h"
#include "ScrollingStateNode.h"
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/TypeCasts.h>

namespace WebCore {

class ScrollingStateFixedNode;
class ScrollingTree;
class ScrollingTreeFrameScrollingNode;
class ScrollingTreeScrollingNode;

class ScrollingTreeNode : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<ScrollingTreeNode> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ScrollingTreeNode, WEBCORE_EXPORT);
    friend class ScrollingTree;
public:
    virtual ~ScrollingTreeNode();

    ScrollingNodeType nodeType() const { return m_nodeType; }
    ScrollingNodeID scrollingNodeID() const { return m_nodeID; }
    
    bool isFixedNode() const { return nodeType() == ScrollingNodeType::Fixed; }
    bool isStickyNode() const { return nodeType() == ScrollingNodeType::Sticky; }
    bool isPositionedNode() const { return nodeType() == ScrollingNodeType::Positioned; }
#if PLATFORM(COCOA)
    bool isFixedNodeCocoa() const { return isFixedNode(); }
    bool isPositionedNodeCocoa() const { return isPositionedNode(); }
    bool isOverflowScrollProxyNodeCocoa() const { return isOverflowScrollProxyNode(); }
#endif
#if USE(COORDINATED_GRAPHICS)
    bool isFixedNodeCoordinated() const { return isFixedNode(); }
    bool isPositionedNodeCoordinated() const { return isPositionedNode(); }
    bool isOverflowScrollProxyNodeCoordinated() const { return isOverflowScrollProxyNode(); }
#endif
    bool isScrollingNode() const { return isFrameScrollingNode() || isOverflowScrollingNode() || isPluginScrollingNode(); }
    bool isFrameScrollingNode() const { return nodeType() == ScrollingNodeType::MainFrame || nodeType() == ScrollingNodeType::Subframe; }
    bool isFrameHostingNode() const { return nodeType() == ScrollingNodeType::FrameHosting; }
    bool isPluginScrollingNode() const { return nodeType() == ScrollingNodeType::PluginScrolling; }
    bool isPluginHostingNode() const { return nodeType() == ScrollingNodeType::PluginHosting; }
    bool isOverflowScrollingNode() const { return nodeType() == ScrollingNodeType::Overflow; }
    bool isOverflowScrollProxyNode() const { return nodeType() == ScrollingNodeType::OverflowProxy; }

    virtual bool commitStateBeforeChildren(const ScrollingStateNode&) = 0;
    virtual bool commitStateAfterChildren(const ScrollingStateNode&) { return true; }
    virtual void didCompleteCommitForNode() { }
    
    virtual void willBeDestroyed() { }

    RefPtr<ScrollingTreeNode> parent() const { return m_parent.get(); }
    void setParent(RefPtr<ScrollingTreeNode>&& parent) { m_parent = parent; }

    WEBCORE_EXPORT bool isRootNode() const;

    const Vector<Ref<ScrollingTreeNode>>& children() const { return m_children; }

    void appendChild(Ref<ScrollingTreeNode>&&);
    void removeChild(ScrollingTreeNode&);
    void removeAllChildren();

    virtual bool isRootOfHostedSubtree() const { return false; }

    WEBCORE_EXPORT RefPtr<ScrollingTreeFrameScrollingNode> enclosingFrameNodeIncludingSelf();
    WEBCORE_EXPORT RefPtr<ScrollingTreeScrollingNode> enclosingScrollingNodeIncludingSelf();

    WEBCORE_EXPORT void dump(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const;

    FrameIdentifier frameIdentifier() const { return *m_parentFrameIdentifier; }
    void setFrameIdentifier(FrameIdentifier frameID) { m_parentFrameIdentifier = frameID; }

protected:
    ScrollingTreeNode(ScrollingTree&, ScrollingNodeType, ScrollingNodeID);
    RefPtr<ScrollingTree> scrollingTree() const { return m_scrollingTree.get(); }

    virtual void applyLayerPositions() = 0;

    WEBCORE_EXPORT virtual void dumpProperties(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const;

    Vector<Ref<ScrollingTreeNode>> m_children;

private:
    ThreadSafeWeakPtr<ScrollingTree> m_scrollingTree;

    const ScrollingNodeType m_nodeType;
    const ScrollingNodeID m_nodeID;
    Markable<FrameIdentifier> m_parentFrameIdentifier;

    ThreadSafeWeakPtr<ScrollingTreeNode> m_parent;
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_SCROLLING_NODE(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToValueTypeName) \
    static bool isType(const WebCore::ScrollingTreeNode& node) { return node.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(ASYNC_SCROLLING)
