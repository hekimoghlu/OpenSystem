/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 22, 2022.
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

#include <wtf/CheckedPtr.h>

#if ASSERT_ENABLED
#include "ElementIteratorAssertions.h"
#endif

namespace WebCore {

class ContainerNode;
class Node;

template <typename ElementType>
class ElementIterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = ElementType;
    using difference_type = ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;

    ElementIterator() = default;

    inline ElementType& operator*() const;
    inline ElementType* operator->() const;

    constexpr operator bool() const { return m_current.get(); }
    constexpr bool operator!() const { return !m_current; }
    constexpr bool operator==(std::nullptr_t) const { return !m_current; }
    constexpr bool operator==(const ElementIterator&) const;

    inline ElementIterator& traverseNext();
    inline ElementIterator& traversePrevious();
    inline ElementIterator& traverseNextSibling();
    inline ElementIterator& traversePreviousSibling();
    inline ElementIterator& traverseNextSkippingChildren();
    inline ElementIterator& traverseAncestor();

    inline void dropAssertions();

protected:
    ElementIterator(const ContainerNode* root, ElementType* current);

private:
    CheckedPtr<const ContainerNode> m_root;
    CheckedPtr<ElementType> m_current;

#if ASSERT_ENABLED
    ElementIteratorAssertions m_assertions;
#endif
};

template <typename ElementType>
inline ElementType* findElementAncestorOfType(const Node&);

// ElementIterator

template <typename ElementType>
inline ElementIterator<ElementType>::ElementIterator(const ContainerNode* root, ElementType* current)
    : m_root(root)
    , m_current(current)
#if ASSERT_ENABLED
    , m_assertions(current)
#endif
{
}

template<typename ElementType> constexpr bool ElementIterator<ElementType>::operator==(const ElementIterator& other) const
{
    ASSERT(m_root == other.m_root || !m_current || !other.m_current);
    return m_current == other.m_current;
}

template <typename ElementType>
inline ElementType& ElementIterator<ElementType>::operator*() const
{
    ASSERT(m_current);
    ASSERT(!m_assertions.domTreeHasMutated());
    return *m_current;
}

template <typename ElementType>
inline ElementType* ElementIterator<ElementType>::operator->() const
{
    ASSERT(m_current);
    ASSERT(!m_assertions.domTreeHasMutated());
    return m_current.get();
}

template <typename ElementType>
inline void ElementIterator<ElementType>::dropAssertions()
{
#if ASSERT_ENABLED
    m_assertions.clear();
#endif
}

} // namespace WebCore
