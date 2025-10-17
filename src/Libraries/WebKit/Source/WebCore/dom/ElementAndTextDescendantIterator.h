/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 25, 2023.
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

#include "Element.h"
#include "ElementIteratorAssertions.h"
#include "Text.h"
#include <wtf/Vector.h>

namespace WebCore {

class ElementAndTextDescendantIterator {
public:
    ElementAndTextDescendantIterator() = default;
    enum FirstChildTag { FirstChild };
    ElementAndTextDescendantIterator(const ContainerNode& root, FirstChildTag);
    ElementAndTextDescendantIterator(const ContainerNode& root, Node* current);

    ElementAndTextDescendantIterator& operator++() { return traverseNext(); }

    Node& operator*();
    Node* operator->();
    const Node& operator*() const;
    const Node* operator->() const;

    bool operator==(const ElementAndTextDescendantIterator& other) const;

    bool operator!() const { return !m_depth; }
    explicit operator bool() const { return m_depth; }

    void dropAssertions();

    ElementAndTextDescendantIterator& traverseNext();
    ElementAndTextDescendantIterator& traverseNextSkippingChildren();
    ElementAndTextDescendantIterator& traverseNextSibling();
    ElementAndTextDescendantIterator& traversePreviousSibling();

    unsigned depth() const { return m_depth; }

private:
    static bool isElementOrText(const Node& node) { return is<Element>(node) || is<Text>(node); }
    static Node* firstChild(const Node&);
    static Node* nextSibling(const Node&);
    static Node* previousSibling(const Node&);

    void popAncestorSiblingStack();

    CheckedPtr<Node> m_current;
    struct AncestorSibling {
        CheckedPtr<Node> node;
        unsigned depth;
    };
    Vector<AncestorSibling, 16> m_ancestorSiblingStack;
    unsigned m_depth { 0 };

#if ASSERT_ENABLED
    ElementIteratorAssertions m_assertions;
#endif
};

class ElementAndTextDescendantRange {
public:
    explicit ElementAndTextDescendantRange(const ContainerNode& root);
    ElementAndTextDescendantIterator begin() const;
    ElementAndTextDescendantIterator end() const;

private:
    CheckedRef<const ContainerNode> m_root;
};

ElementAndTextDescendantRange elementAndTextDescendants(ContainerNode&);

// ElementAndTextDescendantIterator

inline ElementAndTextDescendantIterator::ElementAndTextDescendantIterator(const ContainerNode& root, FirstChildTag)
    : m_current(firstChild(root))
#if ASSERT_ENABLED
    , m_assertions(m_current.get())
#endif
{
    if (!m_current)
        return;
    m_ancestorSiblingStack.append({ nullptr, 0 });
    m_depth = 1;
}

inline ElementAndTextDescendantIterator::ElementAndTextDescendantIterator(const ContainerNode& root, Node* current)
    : m_current(current)
#if ASSERT_ENABLED
    , m_assertions(m_current.get())
#endif
{
    if (!m_current)
        return;
    ASSERT(isElementOrText(*m_current));
    if (m_current == &root)
        return;

    Vector<Node*, 20> ancestorStack;
    auto* ancestor = m_current->parentNode();
    while (ancestor != &root) {
        ancestorStack.append(ancestor);
        ancestor = ancestor->parentNode();
    }

    m_ancestorSiblingStack.append({ nullptr, 0 });
    for (unsigned i = ancestorStack.size(); i; --i) {
        if (auto* sibling = nextSibling(*ancestorStack[i - 1]))
            m_ancestorSiblingStack.append({ sibling, i });
    }

    m_depth = ancestorStack.size() + 1;
}

inline void ElementAndTextDescendantIterator::dropAssertions()
{
#if ASSERT_ENABLED
    m_assertions.clear();
#endif
}

inline Node* ElementAndTextDescendantIterator::firstChild(const Node& current)
{
    auto* node = current.firstChild();
    while (node && !isElementOrText(*node))
        node = node->nextSibling();
    return node;
}

inline Node* ElementAndTextDescendantIterator::nextSibling(const Node& current)
{
    auto* node = current.nextSibling();
    while (node && !isElementOrText(*node))
        node = node->nextSibling();
    return node;
}

inline Node* ElementAndTextDescendantIterator::previousSibling(const Node& current)
{
    auto* node = current.previousSibling();
    while (node && !isElementOrText(*node))
        node = node->previousSibling();
    return node;
}

inline void ElementAndTextDescendantIterator::popAncestorSiblingStack()
{
    m_current = m_ancestorSiblingStack.last().node;
    m_depth = m_ancestorSiblingStack.last().depth;
    m_ancestorSiblingStack.removeLast();

#if ASSERT_ENABLED
    // Drop the assertion when the iterator reaches the end.
    if (!m_current)
        m_assertions.dropEventDispatchAssertion();
#endif
}

inline ElementAndTextDescendantIterator& ElementAndTextDescendantIterator::traverseNext()
{
    ASSERT(m_current);
    ASSERT(!m_assertions.domTreeHasMutated());

    auto* firstChild = ElementAndTextDescendantIterator::firstChild(*m_current);
    auto* nextSibling = ElementAndTextDescendantIterator::nextSibling(*m_current);
    if (firstChild) {
        if (nextSibling)
            m_ancestorSiblingStack.append({ nextSibling, m_depth });
        ++m_depth;
        m_current = firstChild;
        return *this;
    }
    if (!nextSibling) {
        popAncestorSiblingStack();
        return *this;
    }

    m_current = nextSibling;
    return *this;
}

inline ElementAndTextDescendantIterator& ElementAndTextDescendantIterator::traverseNextSkippingChildren()
{
    ASSERT(m_current);
    ASSERT(!m_assertions.domTreeHasMutated());

    auto* nextSibling = ElementAndTextDescendantIterator::nextSibling(*m_current);
    if (!nextSibling) {
        popAncestorSiblingStack();
        return *this;
    }

    m_current = nextSibling;
    return *this;
}

inline ElementAndTextDescendantIterator& ElementAndTextDescendantIterator::traverseNextSibling()
{
    ASSERT(m_current);
    ASSERT(!m_assertions.domTreeHasMutated());

    m_current = nextSibling(*m_current);

#if ASSERT_ENABLED
    if (!m_current)
        m_assertions.dropEventDispatchAssertion();
#endif
    return *this;
}

inline ElementAndTextDescendantIterator& ElementAndTextDescendantIterator::traversePreviousSibling()
{
    ASSERT(m_current);
    ASSERT(!m_assertions.domTreeHasMutated());

    m_current = previousSibling(*m_current);

#if ASSERT_ENABLED
    if (!m_current)
        m_assertions.dropEventDispatchAssertion();
#endif
    return *this;
}

inline Node& ElementAndTextDescendantIterator::operator*()
{
    ASSERT(m_current);
    ASSERT(isElementOrText(*m_current));
    ASSERT(!m_assertions.domTreeHasMutated());
    return *m_current;
}

inline Node* ElementAndTextDescendantIterator::operator->()
{
    ASSERT(m_current);
    ASSERT(isElementOrText(*m_current));
    ASSERT(!m_assertions.domTreeHasMutated());
    return m_current.get();
}

inline const Node& ElementAndTextDescendantIterator::operator*() const
{
    ASSERT(m_current);
    ASSERT(isElementOrText(*m_current));
    ASSERT(!m_assertions.domTreeHasMutated());
    return *m_current;
}

inline const Node* ElementAndTextDescendantIterator::operator->() const
{
    ASSERT(m_current);
    ASSERT(isElementOrText(*m_current));
    ASSERT(!m_assertions.domTreeHasMutated());
    return m_current.get();
}

inline bool ElementAndTextDescendantIterator::operator==(const ElementAndTextDescendantIterator& other) const
{
    ASSERT(!m_assertions.domTreeHasMutated());
    return m_current == other.m_current || (!m_depth && !other.m_depth);
}

// ElementAndTextDescendantRange

inline ElementAndTextDescendantRange::ElementAndTextDescendantRange(const ContainerNode& root)
    : m_root(root)
{
}

inline ElementAndTextDescendantIterator ElementAndTextDescendantRange::begin() const
{
    return ElementAndTextDescendantIterator(m_root.get(), ElementAndTextDescendantIterator::FirstChild);
}

inline ElementAndTextDescendantIterator ElementAndTextDescendantRange::end() const
{
    return { };
}

// Standalone functions

inline ElementAndTextDescendantRange elementAndTextDescendants(ContainerNode& root)
{
    return ElementAndTextDescendantRange(root);
}

} // namespace WebCore
