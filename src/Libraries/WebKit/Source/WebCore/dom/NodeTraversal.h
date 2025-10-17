/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 29, 2024.
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
#include "Text.h"

namespace WebCore {
namespace NodeTraversal {

// Does a pre-order traversal of the tree to find the next node after this one.
// This uses the same order that tags appear in the source file. If the stayWithin
// argument is non-null, the traversal will stop once the specified node is reached.
// This can be used to restrict traversal to a particular sub-tree.
Node* next(const Node&);
Node* next(const Node&, const Node* stayWithin);
Node* next(const ContainerNode&);
Node* next(const ContainerNode&, const Node* stayWithin);
Node* next(const Text&);
Node* next(const Text&, const Node* stayWithin);

// Like next, but skips children and starts with the next sibling.
Node* nextSkippingChildren(const Node&);
Node* nextSkippingChildren(const Node&, const Node* stayWithin);

// Does a reverse pre-order traversal to find the node that comes before the current one in document order
WEBCORE_EXPORT Node* last(const ContainerNode&);
Node* previous(const Node&, const Node* stayWithin = nullptr);

// Like previous, but skips children and starts with the next sibling.
Node* previousSkippingChildren(const Node&, const Node* stayWithin = nullptr);

// Like next, but visits parents after their children.
Node* nextPostOrder(const Node&, const Node* stayWithin = nullptr);

// Like previous/previousSkippingChildren, but visits parents before their children.
Node* previousPostOrder(const Node&, const Node* stayWithin = nullptr);
Node* previousSkippingChildrenPostOrder(const Node&, const Node* stayWithin = nullptr);

// Pre-order traversal including the pseudo-elements.
Node* previousIncludingPseudo(const Node&, const Node* = nullptr);
Node* nextIncludingPseudo(const Node&, const Node* = nullptr);
Node* nextIncludingPseudoSkippingChildren(const Node&, const Node* = nullptr);

}

namespace NodeTraversal {

WEBCORE_EXPORT Node* nextAncestorSibling(const Node&);
WEBCORE_EXPORT Node* nextAncestorSibling(const Node&, const Node* stayWithin);
WEBCORE_EXPORT Node* deepLastChild(Node&);

template <class NodeType>
inline Node* traverseNextTemplate(NodeType& current)
{
    if (current.firstChild())
        return current.firstChild();
    if (current.nextSibling())
        return current.nextSibling();
    return nextAncestorSibling(current);
}
inline Node* next(const Node& current) { return traverseNextTemplate(current); }
inline Node* next(const ContainerNode& current) { return traverseNextTemplate(current); }

template <class NodeType>
inline Node* traverseNextTemplate(NodeType& current, const Node* stayWithin)
{
    if (current.firstChild())
        return current.firstChild();
    if (&current == stayWithin)
        return nullptr;
    if (current.nextSibling())
        return current.nextSibling();
    return nextAncestorSibling(current, stayWithin);
}
inline Node* next(const Node& current, const Node* stayWithin) { return traverseNextTemplate(current, stayWithin); }
inline Node* next(const ContainerNode& current, const Node* stayWithin) { return traverseNextTemplate(current, stayWithin); }

inline Node* nextSkippingChildren(const Node& current)
{
    if (current.nextSibling())
        return current.nextSibling();
    return nextAncestorSibling(current);
}

inline Node* nextSkippingChildren(const Node& current, const Node* stayWithin)
{
    if (&current == stayWithin)
        return nullptr;
    if (current.nextSibling())
        return current.nextSibling();
    return nextAncestorSibling(current, stayWithin);
}

inline Node* next(const Text& current) { return nextSkippingChildren(current); }
inline Node* next(const Text& current, const Node* stayWithin) { return nextSkippingChildren(current, stayWithin); }

inline Node* previous(const Node& current, const Node* stayWithin)
{
    if (Node* previous = current.previousSibling())
        return deepLastChild(*previous);
    if (current.parentNode() == stayWithin)
        return nullptr;
    return current.parentNode();
}

}
}
