/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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
#include "NodeTraversal.h"

#include "PseudoElement.h"

namespace WebCore {

namespace NodeTraversal {

Node* previousIncludingPseudo(const Node& current, const Node* stayWithin)
{
    Node* previous;
    if (&current == stayWithin)
        return nullptr;
    if ((previous = current.pseudoAwarePreviousSibling())) {
        while (previous->pseudoAwareLastChild())
            previous = previous->pseudoAwareLastChild();
        return previous;
    }
    auto* pseudoElement = dynamicDowncast<PseudoElement>(current);
    return pseudoElement ? pseudoElement->hostElement() : current.parentNode();
}

Node* nextIncludingPseudo(const Node& current, const Node* stayWithin)
{
    Node* next;
    if ((next = current.pseudoAwareFirstChild()))
        return next;
    if (&current == stayWithin)
        return nullptr;
    if ((next = current.pseudoAwareNextSibling()))
        return next;
    auto* pseudoElement = dynamicDowncast<PseudoElement>(current);
    const Node* ancestor = pseudoElement ? pseudoElement->hostElement() : current.parentNode();
    for (; ancestor; ancestor = ancestor->parentNode()) {
        if (ancestor == stayWithin)
            return nullptr;
        if ((next = ancestor->pseudoAwareNextSibling()))
            return next;
    }
    return nullptr;
}

Node* nextIncludingPseudoSkippingChildren(const Node& current, const Node* stayWithin)
{
    Node* next;
    if (&current == stayWithin)
        return nullptr;
    if ((next = current.pseudoAwareNextSibling()))
        return next;
    auto* pseudoElement = dynamicDowncast<PseudoElement>(current);
    const Node* ancestor = pseudoElement ? pseudoElement->hostElement() : current.parentNode();
    for (; ancestor; ancestor = ancestor->parentNode()) {
        if (ancestor == stayWithin)
            return nullptr;
        if ((next = ancestor->pseudoAwareNextSibling()))
            return next;
    }
    return nullptr;
}

Node* nextAncestorSibling(const Node& current)
{
    ASSERT(!current.nextSibling());
    for (auto* ancestor = current.parentNode(); ancestor; ancestor = ancestor->parentNode()) {
        if (ancestor->nextSibling())
            return ancestor->nextSibling();
    }
    return nullptr;
}

Node* nextAncestorSibling(const Node& current, const Node* stayWithin)
{
    ASSERT(!current.nextSibling());
    ASSERT(&current != stayWithin);
    for (auto* ancestor = current.parentNode(); ancestor; ancestor = ancestor->parentNode()) {
        if (ancestor == stayWithin)
            return nullptr;
        if (ancestor->nextSibling())
            return ancestor->nextSibling();
    }
    return nullptr;
}

Node* last(const ContainerNode& current)
{
    Node* node = current.lastChild();
    if (!node)
        return nullptr;
    while (node->lastChild())
        node = node->lastChild();
    return node;
}

Node* deepLastChild(Node& node)
{
    Node* lastChild = &node;
    while (lastChild->lastChild())
        lastChild = lastChild->lastChild();
    return lastChild;
}

Node* previousSkippingChildren(const Node& current, const Node* stayWithin)
{
    if (&current == stayWithin)
        return nullptr;
    if (current.previousSibling())
        return current.previousSibling();
    for (auto* ancestor = current.parentNode(); ancestor; ancestor = ancestor->parentNode()) {
        if (ancestor == stayWithin)
            return nullptr;
        if (ancestor->previousSibling())
            return ancestor->previousSibling();
    }
    return nullptr;
}

Node* nextPostOrder(const Node& current, const Node* stayWithin)
{
    if (&current == stayWithin)
        return nullptr;
    if (!current.nextSibling())
        return current.parentNode();
    Node* next = current.nextSibling();
    while (next->firstChild())
        next = next->firstChild();
    return next;
}

static Node* previousAncestorSiblingPostOrder(const Node& current, const Node* stayWithin)
{
    ASSERT(!current.previousSibling());
    for (auto* ancestor = current.parentNode(); ancestor; ancestor = ancestor->parentNode()) {
        if (ancestor == stayWithin)
            return nullptr;
        if (ancestor->previousSibling())
            return ancestor->previousSibling();
    }
    return nullptr;
}

Node* previousPostOrder(const Node& current, const Node* stayWithin)
{
    if (current.lastChild())
        return current.lastChild();
    if (&current == stayWithin)
        return nullptr;
    if (current.previousSibling())
        return current.previousSibling();
    return previousAncestorSiblingPostOrder(current, stayWithin);
}

Node* previousSkippingChildrenPostOrder(const Node& current, const Node* stayWithin)
{
    if (&current == stayWithin)
        return nullptr;
    if (current.previousSibling())
        return current.previousSibling();
    return previousAncestorSiblingPostOrder(current, stayWithin);
}

}
}
