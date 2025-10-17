/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 1, 2024.
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
#include "XPathNodeSet.h"

#include "Attr.h"
#include "Attribute.h"
#include "Document.h"
#include "ElementInlines.h"
#include "NodeTraversal.h"

namespace WebCore {
namespace XPath {

// When a node set is large, sorting it by traversing the whole document is better (we can
// assume that we aren't dealing with documents that we cannot even traverse in reasonable time).
const unsigned traversalSortCutoff = 10000;

static inline Node* parentWithDepth(unsigned depth, const Vector<Node*>& parents)
{
    ASSERT(parents.size() >= depth + 1);
    return parents[parents.size() - 1 - depth];
}

static void sortBlock(unsigned from, unsigned to, Vector<Vector<Node*>>& parentMatrix, bool mayContainAttributeNodes)
{
    ASSERT(from + 1 < to); // Should not call this function with less that two nodes to sort.
    unsigned minDepth = UINT_MAX;
    for (unsigned i = from; i < to; ++i) {
        unsigned depth = parentMatrix[i].size() - 1;
        if (minDepth > depth)
            minDepth = depth;
    }
    
    // Find the common ancestor.
    unsigned commonAncestorDepth = minDepth;
    Node* commonAncestor;
    while (true) {
        commonAncestor = parentWithDepth(commonAncestorDepth, parentMatrix[from]);
        if (commonAncestorDepth == 0)
            break;

        bool allEqual = true;
        for (unsigned i = from + 1; i < to; ++i) {
            if (commonAncestor != parentWithDepth(commonAncestorDepth, parentMatrix[i])) {
                allEqual = false;
                break;
            }
        }
        if (allEqual)
            break;
        
        --commonAncestorDepth;
    }

    if (commonAncestorDepth == minDepth) {
        // One of the nodes is the common ancestor => it is the first in document order.
        // Find it and move it to the beginning.
        for (unsigned i = from; i < to; ++i)
            if (commonAncestor == parentMatrix[i][0]) {
                parentMatrix[i].swap(parentMatrix[from]);
                if (from + 2 < to)
                    sortBlock(from + 1, to, parentMatrix, mayContainAttributeNodes);
                return;
            }
    }
    
    if (mayContainAttributeNodes && commonAncestor->isElementNode()) {
        // The attribute nodes and namespace nodes of an element occur before the children of the element.
        // The namespace nodes are defined to occur before the attribute nodes.
        // The relative order of namespace nodes is implementation-dependent.
        // The relative order of attribute nodes is implementation-dependent.
        unsigned sortedEnd = from;
        // FIXME: namespace nodes are not implemented.
        for (unsigned i = sortedEnd; i < to; ++i) {
            auto* node = parentMatrix[i][0];
            if (auto* attr = dynamicDowncast<Attr>(*node); attr && attr->ownerElement() == commonAncestor)
                parentMatrix[i].swap(parentMatrix[sortedEnd++]);
        }
        if (sortedEnd != from) {
            if (to - sortedEnd > 1)
                sortBlock(sortedEnd, to, parentMatrix, mayContainAttributeNodes);
            return;
        }
    }

    // Children nodes of the common ancestor induce a subdivision of our node-set.
    // Sort it according to this subdivision, and recursively sort each group.
    HashSet<RefPtr<Node>> parentNodes;
    for (unsigned i = from; i < to; ++i)
        parentNodes.add(parentWithDepth(commonAncestorDepth + 1, parentMatrix[i]));

    unsigned previousGroupEnd = from;
    unsigned groupEnd = from;
    for (Node* n = commonAncestor->firstChild(); n; n = n->nextSibling()) {
        // If parentNodes contains the node, perform a linear search to move its children in the node-set to the beginning.
        if (parentNodes.contains(n)) {
            for (unsigned i = groupEnd; i < to; ++i)
                if (parentWithDepth(commonAncestorDepth + 1, parentMatrix[i]) == n)
                    parentMatrix[i].swap(parentMatrix[groupEnd++]);

            if (groupEnd - previousGroupEnd > 1)
                sortBlock(previousGroupEnd, groupEnd, parentMatrix, mayContainAttributeNodes);

            ASSERT(previousGroupEnd != groupEnd);
            previousGroupEnd = groupEnd;
#if ASSERT_ENABLED
            parentNodes.remove(n);
#endif
        }
    }

    ASSERT(parentNodes.isEmpty());
}

void NodeSet::sort() const
{
    if (m_isSorted)
        return;

    unsigned nodeCount = m_nodes.size();
    if (nodeCount < 2) {
        m_isSorted = true;
        return;
    }

    if (nodeCount > traversalSortCutoff) {
        traversalSort();
        return;
    }

    bool containsAttributeNodes = false;
    
    Vector<Vector<Node*>> parentMatrix(nodeCount);
    for (unsigned i = 0; i < nodeCount; ++i) {
        Vector<Node*>& parentsVector = parentMatrix[i];
        auto* node = m_nodes[i].get();
        parentsVector.append(node);
        if (auto* attr = dynamicDowncast<Attr>(*node)) {
            node = attr->ownerElement();
            parentsVector.append(node);
            containsAttributeNodes = true;
        }
        while ((node = node->parentNode()))
            parentsVector.append(node);
    }
    sortBlock(0, nodeCount, parentMatrix, containsAttributeNodes);
    
    // It is not possible to just assign the result to m_nodes, because some nodes may get dereferenced and destroyed.
    Vector<RefPtr<Node>> sortedNodes;
    sortedNodes.reserveInitialCapacity(nodeCount);
    for (unsigned i = 0; i < nodeCount; ++i)
        sortedNodes.append(parentMatrix[i][0]);
    
    m_nodes = WTFMove(sortedNodes);
    m_isSorted = true;
}

static Node* findRootNode(Node* node)
{
    if (auto* attr = dynamicDowncast<Attr>(*node))
        node = attr->ownerElement();
    if (node->isConnected())
        node = &node->document();
    else {
        while (Node* parent = node->parentNode())
            node = parent;
    }
    return node;
}

void NodeSet::traversalSort() const
{
    HashSet<RefPtr<Node>> nodes;
    bool containsAttributeNodes = false;

    unsigned nodeCount = m_nodes.size();
    ASSERT(nodeCount > 1);
    for (auto& node : m_nodes) {
        nodes.add(node.get());
        if (node->isAttributeNode())
            containsAttributeNodes = true;
    }

    Vector<RefPtr<Node>> sortedNodes;
    sortedNodes.reserveInitialCapacity(nodeCount);

    for (Node* node = findRootNode(m_nodes.first().get()); node; node = NodeTraversal::next(*node)) {
        if (nodes.contains(node))
            sortedNodes.append(node);

        if (!containsAttributeNodes)
            continue;

        RefPtr element = dynamicDowncast<Element>(*node);
        if (!element || !element->hasAttributes())
            continue;

        for (auto& attribute : element->attributes()) {
            RefPtr attr = element->attrIfExists(attribute.name());
            if (attr && nodes.contains(attr.get()))
                sortedNodes.append(attr);
        }
    }

    ASSERT(sortedNodes.size() == nodeCount);
    m_nodes = WTFMove(sortedNodes);
    m_isSorted = true;
}

Node* NodeSet::firstNode() const
{
    if (isEmpty())
        return nullptr;

    sort(); // FIXME: fully sorting the node-set just to find its first node is wasteful.
    return m_nodes.at(0).get();
}

Node* NodeSet::anyNode() const
{
    if (isEmpty())
        return nullptr;

    return m_nodes.at(0).get();
}

}
}
