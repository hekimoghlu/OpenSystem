/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 1, 2024.
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

#include "ContainerNode.h"
#include <wtf/Assertions.h>

namespace WebCore {

// FIXME: Delete this class after fixing FormListedElement to avoid calling getElementById during a tree removal.
#if ASSERT_ENABLED
class ContainerChildRemovalScope {
public:
    ContainerChildRemovalScope(ContainerNode& parentOfRemovedTree, Node& child)
        : m_parentOfRemovedTree(parentOfRemovedTree)
        , m_removedChild(child)
        , m_previousScope(s_scope)
    {
        s_scope = this;
    }

    ~ContainerChildRemovalScope()
    {
        s_scope = m_previousScope;
    }

    ContainerNode& parentOfRemovedTree() { return m_parentOfRemovedTree; }
    Node& removedChild() { return m_removedChild; }

    static ContainerChildRemovalScope* currentScope() { return s_scope; }

private:
    ContainerNode& m_parentOfRemovedTree;
    Node& m_removedChild;
    ContainerChildRemovalScope* m_previousScope;
    static ContainerChildRemovalScope* s_scope;
};
#else // not ASSERT_ENABLED
class ContainerChildRemovalScope {
public:
    ContainerChildRemovalScope(ContainerNode&, Node&) { }
};
#endif // not ASSERT_ENABLED

void notifyChildNodeInserted(ContainerNode& parentOfInsertedTree, Node&, NodeVector& postInsertionNotificationTargets);
inline void updateCanDelayNodeDeletion(ContainerNode::CanDelayNodeDeletion& currentCanDelayDeletion, ContainerNode::CanDelayNodeDeletion newStatus);

enum class RemovedSubtreeObservability : bool {
    NotObservable,
    MaybeObservableByRefPtr,
};

struct RemovedSubtreeResult {
    unsigned subTreeSize;
    RemovedSubtreeObservability removedSubtreeObservability;
    ContainerNode::CanDelayNodeDeletion canBeDelayed;
};

RemovedSubtreeResult notifyChildNodeRemoved(ContainerNode& oldParentOfRemovedTree, Node&);
void removeDetachedChildrenInContainer(ContainerNode&);

enum class SubframeDisconnectPolicy : bool {
    RootAndDescendants,
    DescendantsOnly
};
void disconnectSubframes(ContainerNode& root, SubframeDisconnectPolicy);

inline void disconnectSubframesIfNeeded(ContainerNode& root, SubframeDisconnectPolicy policy)
{
    if (!root.connectedSubframeCount())
        return;
    disconnectSubframes(root, policy);
}

inline void updateCanDelayNodeDeletion(ContainerNode::CanDelayNodeDeletion& currentCanDelayDeletion, ContainerNode::CanDelayNodeDeletion newStatus)
{
    if (newStatus == ContainerNode::CanDelayNodeDeletion::No)
        currentCanDelayDeletion = ContainerNode::CanDelayNodeDeletion::No;
}

} // namespace WebCore
