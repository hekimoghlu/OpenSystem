/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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

#include "ElementRareData.h"
#include "HTMLSlotElement.h"
#include "PseudoElement.h"
#include "ShadowRoot.h"

namespace WebCore {

class HTMLSlotElement;

class ComposedTreeAncestorIterator {
public:
    ComposedTreeAncestorIterator();
    ComposedTreeAncestorIterator(Element& current);
    ComposedTreeAncestorIterator(Node& current);

    Element& operator*() { return get(); }
    Element* operator->() { return &get(); }

    friend bool operator==(ComposedTreeAncestorIterator, ComposedTreeAncestorIterator) = default;

    ComposedTreeAncestorIterator& operator++()
    {
        m_current = traverseParent(m_current.get());
        return *this;
    }

    Element& get() { return *m_current; }

private:
    void traverseParentInShadowTree();
    static Element* traverseParent(Node*);

    CheckedPtr<Element> m_current;
};

inline ComposedTreeAncestorIterator::ComposedTreeAncestorIterator()
{
}

inline ComposedTreeAncestorIterator::ComposedTreeAncestorIterator(Node& current)
    : m_current(traverseParent(&current))
{
    ASSERT(!is<ShadowRoot>(current));
}

inline ComposedTreeAncestorIterator::ComposedTreeAncestorIterator(Element& current)
    : m_current(&current)
{
}

inline Element* ComposedTreeAncestorIterator::traverseParent(Node* current)
{
    auto* parent = current->parentNode();
    if (!parent)
        return nullptr;
    if (auto* shadowRoot = dynamicDowncast<ShadowRoot>(*parent))
        return shadowRoot->host();
    auto* parentElement = dynamicDowncast<Element>(*parent);
    if (!parentElement)
        return nullptr;
    if (auto* shadowRoot = parentElement->shadowRoot())
        return shadowRoot->findAssignedSlot(*current);
    return parentElement;
}

class ComposedTreeAncestorAdapter {
public:
    using iterator = ComposedTreeAncestorIterator;

    ComposedTreeAncestorAdapter(Node& node)
        : m_node(node)
    { }

    iterator begin()
    {
        if (auto shadowRoot = dynamicDowncast<ShadowRoot>(m_node.get()))
            return iterator(*shadowRoot->host());
        if (auto pseudoElement = dynamicDowncast<PseudoElement>(m_node.get()))
            return iterator(*pseudoElement->hostElement());
        return iterator(m_node);
    }
    iterator end()
    {
        return iterator();
    }
    Element* first()
    {
        auto it = begin();
        if (it == end())
            return nullptr;
        return &it.get();
    }

private:
    Ref<Node> m_node;
};

// FIXME: We should have const versions too.
inline ComposedTreeAncestorAdapter composedTreeAncestors(Node& node)
{
    return ComposedTreeAncestorAdapter(node);
}

} // namespace WebCore
