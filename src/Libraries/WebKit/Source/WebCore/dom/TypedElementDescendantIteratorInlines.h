/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 20, 2023.
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

#include "ElementIteratorInlines.h"
#include "TypedElementDescendantIterator.h"

namespace WebCore {

// ElementDescendantIterator

template<typename ElementType> ElementDescendantIterator<ElementType>& ElementDescendantIterator<ElementType>::operator++()
{
    ElementIterator<ElementType>::traverseNext();
    return *this;
}

template<typename ElementType> ElementDescendantIterator<ElementType>& ElementDescendantIterator<ElementType>::operator--()
{
    ElementIterator<ElementType>::traversePrevious();
    return *this;
}

// ElementDescendantRange

template<typename ElementType> ElementDescendantIterator<ElementType> ElementDescendantRange<ElementType>::begin() const
{
    return ElementDescendantIterator<ElementType>(m_root, Traversal<ElementType>::firstWithin(m_root));
}

template<typename ElementType> ElementDescendantIterator<ElementType> ElementDescendantRange<ElementType>::beginAt(ElementType& descendant) const
{
    ASSERT(descendant.isDescendantOf(m_root));
    return ElementDescendantIterator<ElementType>(m_root, &descendant);
}

template<typename ElementType> ElementDescendantIterator<ElementType> ElementDescendantRange<ElementType>::from(Element& descendant) const
{
    ASSERT(descendant.isDescendantOf(m_root));
    if (auto descendantElement = dynamicDowncast<ElementType>(descendant))
        return ElementDescendantIterator<ElementType>(m_root, descendantElement);
    ElementType* next = Traversal<ElementType>::next(descendant, m_root.ptr());
    return ElementDescendantIterator<ElementType>(m_root, next);
}

template<typename ElementType> ElementType* ElementDescendantRange<ElementType>::first() const
{
    return Traversal<ElementType>::firstWithin(m_root);
}

template<typename ElementType> ElementType* ElementDescendantRange<ElementType>::last() const
{
    return Traversal<ElementType>::lastWithin(m_root);
}

// InclusiveElementDescendantRange

template<typename ElementType> ElementDescendantIterator<ElementType> InclusiveElementDescendantRange<ElementType>::begin() const
{
    return ElementDescendantIterator<ElementType>(m_root, Traversal<ElementType>::inclusiveFirstWithin(const_cast<ContainerNode&>(m_root.get())));
}

template<typename ElementType> ElementDescendantIterator<ElementType> InclusiveElementDescendantRange<ElementType>::beginAt(ElementType& descendant) const
{
    ASSERT(m_root.ptr() == &descendant || descendant.isDescendantOf(m_root));
    return ElementDescendantIterator<ElementType>(m_root, &descendant);
}

template<typename ElementType> ElementDescendantIterator<ElementType> InclusiveElementDescendantRange<ElementType>::from(Element& descendant) const
{
    ASSERT(m_root.ptr() == &descendant || descendant.isDescendantOf(m_root));
    if (auto descendantElement = dynamicDowncast<ElementType>(descendant))
        return ElementDescendantIterator<ElementType>(m_root, descendantElement);
    ElementType* next = Traversal<ElementType>::next(descendant, &m_root);
    return ElementDescendantIterator<ElementType>(m_root, next);
}

template<typename ElementType> ElementType* InclusiveElementDescendantRange<ElementType>::first() const
{
    return Traversal<ElementType>::inclusiveFirstWithin(m_root);
}

template<typename ElementType> ElementType* InclusiveElementDescendantRange<ElementType>::last() const
{
    return Traversal<ElementType>::inclusiveLastWithin(m_root);
}

// DoubleElementDescendantRange

template<typename ElementType> auto DoubleElementDescendantRange<ElementType>::begin() const -> Iterator
{
    return Iterator(m_pair.first.begin(), m_pair.second.begin());
}

// DoubleElementDescendantIterator

template<typename ElementType> auto DoubleElementDescendantIterator<ElementType>::operator*() const -> ReferenceProxy
{
    return { *m_pair.first, *m_pair.second };
}

template<typename ElementType> constexpr bool DoubleElementDescendantIterator<ElementType>::operator==(std::nullptr_t) const
{
    ASSERT(!m_pair.first == !m_pair.second);
    return !m_pair.first;
}

template<typename ElementType> DoubleElementDescendantIterator<ElementType>& DoubleElementDescendantIterator<ElementType>::operator++()
{
    ++m_pair.first;
    ++m_pair.second;
    return *this;
}

// FilteredElementDescendantIterator

template<typename ElementType, bool filter(const ElementType&)> FilteredElementDescendantIterator<ElementType, filter>& FilteredElementDescendantIterator<ElementType, filter>::operator++()
{
    do {
        ElementIterator<ElementType>::traverseNext();
    } while (*this && !filter(**this));
    return *this;
}

// FilteredElementDescendantRange

template<typename ElementType, bool filter(const ElementType&)> auto FilteredElementDescendantRange<ElementType, filter>::begin() const -> Iterator
{
    return { m_root, first() };
}

template<typename ElementType, bool filter(const ElementType&)> ElementType* FilteredElementDescendantRange<ElementType, filter>::first() const
{
    for (auto* element = Traversal<ElementType>::firstWithin(m_root.get()); element; element = Traversal<ElementType>::next(*element, m_root.ptr())) {
        if (filter(*element))
            return element;
    }
    return nullptr;
}

// Standalone functions

template<typename ElementType> ElementDescendantRange<ElementType> descendantsOfType(ContainerNode& root)
{
    return ElementDescendantRange<ElementType>(root);
}

template<typename ElementType> InclusiveElementDescendantRange<ElementType> inclusiveDescendantsOfType(ContainerNode& root)
{
    return InclusiveElementDescendantRange<ElementType>(root);
}

template<typename ElementType> ElementDescendantRange<const ElementType> descendantsOfType(const ContainerNode& root)
{
    return ElementDescendantRange<const ElementType>(root);
}

template<typename ElementType> DoubleElementDescendantRange<ElementType> descendantsOfType(ContainerNode& firstRoot, ContainerNode& secondRoot)
{
    return { descendantsOfType<ElementType>(firstRoot), descendantsOfType<ElementType>(secondRoot) };
}

template<typename ElementType, bool filter(const ElementType&)> FilteredElementDescendantRange<ElementType, filter> filteredDescendants(const ContainerNode& root)
{
    return { root };
}


}
