/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
#include "NodeIterator.h"

#include "Document.h"
#include "DocumentInlines.h"
#include "NodeTraversal.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(NodeIterator);

inline NodeIterator::NodePointer::NodePointer(Node& node, bool isPointerBeforeNode)
    : node(&node)
    , isPointerBeforeNode(isPointerBeforeNode)
{
}

inline void NodeIterator::NodePointer::clear()
{
    node = nullptr;
}

inline bool NodeIterator::NodePointer::moveToNext(Node& root)
{
    RefPtr currentNode = node;
    if (!currentNode)
        return false;
    if (isPointerBeforeNode) {
        isPointerBeforeNode = false;
        return true;
    }
    node = NodeTraversal::next(*currentNode, &root);
    return node;
}

inline bool NodeIterator::NodePointer::moveToPrevious(Node& root)
{
    RefPtr currentNode = node;
    if (!currentNode)
        return false;
    if (!isPointerBeforeNode) {
        isPointerBeforeNode = true;
        return true;
    }
    if (currentNode == &root) {
        node = nullptr;
        return false;
    }
    node = NodeTraversal::previous(*currentNode);
    return node;
}

inline NodeIterator::NodeIterator(Node& rootNode, unsigned whatToShow, RefPtr<NodeFilter>&& filter)
    : NodeIteratorBase(rootNode, whatToShow, WTFMove(filter))
    , m_referenceNode(rootNode, true)
{
    root().protectedDocument()->attachNodeIterator(*this);
}

Ref<NodeIterator> NodeIterator::create(Node& rootNode, unsigned whatToShow, RefPtr<NodeFilter>&& filter)
{
    return adoptRef(*new NodeIterator(rootNode, whatToShow, WTFMove(filter)));
}

NodeIterator::~NodeIterator()
{
    root().document().detachNodeIterator(*this);
}

ExceptionOr<RefPtr<Node>> NodeIterator::nextNode()
{
    RefPtr<Node> result;

    m_candidateNode = m_referenceNode;
    Ref root = this->root();
    while (m_candidateNode.moveToNext(root)) {
        // NodeIterators treat the DOM tree as a flat list of nodes.
        // In other words, FILTER_REJECT does not pass over descendants
        // of the rejected node. Hence, FILTER_REJECT is the same as FILTER_SKIP.
        RefPtr provisionalResult = m_candidateNode.node;

        auto filterResult = acceptNode(*provisionalResult);
        if (filterResult.hasException()) {
            m_candidateNode.clear();
            return filterResult.releaseException();
        }

        bool nodeWasAccepted = filterResult.returnValue() == NodeFilter::FILTER_ACCEPT;
        if (nodeWasAccepted) {
            m_referenceNode = m_candidateNode;
            result = WTFMove(provisionalResult);
            break;
        }
    }

    m_candidateNode.clear();
    return result;
}

ExceptionOr<RefPtr<Node>> NodeIterator::previousNode()
{
    RefPtr<Node> result;

    m_candidateNode = m_referenceNode;
    Ref root = this->root();
    while (m_candidateNode.moveToPrevious(root)) {
        // NodeIterators treat the DOM tree as a flat list of nodes.
        // In other words, FILTER_REJECT does not pass over descendants
        // of the rejected node. Hence, FILTER_REJECT is the same as FILTER_SKIP.
        RefPtr provisionalResult = m_candidateNode.node;

        auto filterResult = acceptNode(*provisionalResult);
        if (filterResult.hasException()) {
            m_candidateNode.clear();
            return filterResult.releaseException();
        }

        bool nodeWasAccepted = filterResult.returnValue() == NodeFilter::FILTER_ACCEPT;
        if (nodeWasAccepted) {
            m_referenceNode = m_candidateNode;
            result = WTFMove(provisionalResult);
            break;
        }
    }

    m_candidateNode.clear();
    return result;
}

void NodeIterator::nodeWillBeRemoved(Node& removedNode)
{
    updateForNodeRemoval(removedNode, m_candidateNode);
    updateForNodeRemoval(removedNode, m_referenceNode);
}

void NodeIterator::updateForNodeRemoval(Node& removedNode, NodePointer& referenceNode) const
{
    ASSERT(&root().document() == &removedNode.document());

    // Iterator is not affected if the removed node is the reference node and is the root.
    // or if removed node is not the reference node, or the ancestor of the reference node.
    Ref root = this->root();
    if (!removedNode.isDescendantOf(root))
        return;
    bool willRemoveReferenceNode = &removedNode == referenceNode.node;
    bool willRemoveReferenceNodeAncestor = referenceNode.node && referenceNode.node->isDescendantOf(removedNode);
    if (!willRemoveReferenceNode && !willRemoveReferenceNodeAncestor)
        return;

    if (referenceNode.isPointerBeforeNode) {
        RefPtr node = NodeTraversal::next(removedNode, root.ptr());
        if (node) {
            // Move out from under the node being removed if the new reference
            // node is a descendant of the node being removed.
            while (node && node->isDescendantOf(removedNode))
                node = NodeTraversal::next(*node, root.ptr());
            if (node)
                referenceNode.node = node;
        } else {
            node = NodeTraversal::previous(removedNode);
            if (node) {
                // Move out from under the node being removed if the reference node is
                // a descendant of the node being removed.
                if (willRemoveReferenceNodeAncestor) {
                    while (node && node->isDescendantOf(&removedNode))
                        node = NodeTraversal::previous(*node);
                }
                if (node) {
                    // Removing last node.
                    // Need to move the pointer after the node preceding the 
                    // new reference node.
                    referenceNode.node = node;
                    referenceNode.isPointerBeforeNode = false;
                }
            }
        }
    } else {
        RefPtr node = NodeTraversal::previous(removedNode);
        if (node) {
            // Move out from under the node being removed if the reference node is
            // a descendant of the node being removed.
            if (willRemoveReferenceNodeAncestor) {
                while (node && node->isDescendantOf(removedNode))
                    node = NodeTraversal::previous(*node);
            }
            if (node)
                referenceNode.node = WTFMove(node);
        } else {
            // FIXME: This branch doesn't appear to have any LayoutTests.
            node = NodeTraversal::next(removedNode, root.ptr());
            // Move out from under the node being removed if the reference node is
            // a descendant of the node being removed.
            if (willRemoveReferenceNodeAncestor) {
                while (node && node->isDescendantOf(removedNode))
                    node = NodeTraversal::previous(*node);
            }
            if (node)
                referenceNode.node = WTFMove(node);
        }
    }
}

} // namespace WebCore
