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
#pragma once

#include "ElementIterator.h"

namespace WebCore {

class Element;

template<typename> class DoubleElementDescendantIterator;
template<typename> class DoubleElementDescendantRange;
template<typename> class ElementDescendantRange;
template<typename> class InclusiveElementDescendantRange;
template<typename ElementType, bool(const ElementType&)> class FilteredElementDescendantRange;

// Range for iterating through descendant elements.
template<typename ElementType>
inline ElementDescendantRange<ElementType> descendantsOfType(ContainerNode&);
template<typename ElementType>
inline ElementDescendantRange<const ElementType> descendantsOfType(const ContainerNode&);
template<typename ElementType>
inline InclusiveElementDescendantRange<ElementType> inclusiveDescendantsOfType(ContainerNode&);

// Range that skips elements where the filter returns false.
template<typename ElementType, bool filter(const ElementType&)>
inline FilteredElementDescendantRange<ElementType, filter> filteredDescendants(const ContainerNode&);

// Range for use when both sets of descendants are known to be the same length.
// If they are different lengths, this will stop when the shorter one reaches the end, but also an assertion will fail.
template<typename ElementType> DoubleElementDescendantRange<ElementType> descendantsOfType(ContainerNode& firstRoot, ContainerNode& secondRoot);

template<typename ElementType> class ElementDescendantIterator : public ElementIterator<ElementType> {
public:
    ElementDescendantIterator() = default;
    inline ElementDescendantIterator(const ContainerNode& root, ElementType* current);
    inline ElementDescendantIterator& operator++();
    inline ElementDescendantIterator& operator--();
};

template<typename ElementType> class ElementDescendantRange {
public:
    inline ElementDescendantRange(const ContainerNode& root);
    inline ElementDescendantIterator<ElementType> begin() const;
    static constexpr std::nullptr_t end() { return nullptr; }
    inline ElementDescendantIterator<ElementType> beginAt(ElementType&) const;
    inline ElementDescendantIterator<ElementType> from(Element&) const;

    inline ElementType* first() const;
    inline ElementType* last() const;

private:
    CheckedRef<const ContainerNode> m_root;
};

template<typename ElementType> class InclusiveElementDescendantRange {
public:
    inline InclusiveElementDescendantRange(const ContainerNode& root);
    inline ElementDescendantIterator<ElementType> begin() const;
    static constexpr std::nullptr_t end() { return nullptr; }
    inline ElementDescendantIterator<ElementType> beginAt(ElementType&) const;
    inline ElementDescendantIterator<ElementType> from(Element&) const;

    inline ElementType* first() const;
    inline ElementType* last() const;

private:
    CheckedRef<const ContainerNode> m_root;
};

template<typename ElementType> class DoubleElementDescendantRange {
public:
    typedef ElementDescendantRange<ElementType> SingleAdapter;
    typedef DoubleElementDescendantIterator<ElementType> Iterator;

    inline DoubleElementDescendantRange(SingleAdapter&&, SingleAdapter&&);
    inline Iterator begin() const;
    static constexpr std::nullptr_t end() { return nullptr; }

private:
    std::pair<SingleAdapter, SingleAdapter> m_pair;
};

template<typename ElementType> class DoubleElementDescendantIterator {
public:
    typedef ElementDescendantIterator<ElementType> SingleIterator;
    typedef std::pair<ElementType&, ElementType&> ReferenceProxy;

    inline DoubleElementDescendantIterator(SingleIterator&&, SingleIterator&&);
    inline ReferenceProxy operator*() const;
    constexpr bool operator==(std::nullptr_t) const;
    inline DoubleElementDescendantIterator& operator++();

private:
    std::pair<SingleIterator, SingleIterator> m_pair;
};

template<typename ElementType, bool filter(const ElementType&)> class FilteredElementDescendantIterator : public ElementIterator<ElementType> {
public:
    inline FilteredElementDescendantIterator(const ContainerNode&, ElementType* = nullptr);
    inline FilteredElementDescendantIterator& operator++();
};

template<typename ElementType, bool filter(const ElementType&)> class FilteredElementDescendantRange {
public:
    using Iterator = FilteredElementDescendantIterator<ElementType, filter>;

    inline FilteredElementDescendantRange(const ContainerNode&);
    inline Iterator begin() const;
    static constexpr std::nullptr_t end() { return nullptr; }

    inline ElementType* first() const;

private:
    CheckedRef<const ContainerNode> m_root;
};

// ElementDescendantIterator

template<typename ElementType> ElementDescendantIterator<ElementType>::ElementDescendantIterator(const ContainerNode& root, ElementType* current)
    : ElementIterator<ElementType>(&root, current)
{
}

// ElementDescendantRange

template<typename ElementType> ElementDescendantRange<ElementType>::ElementDescendantRange(const ContainerNode& root)
    : m_root(root)
{
}

// InclusiveElementDescendantRange

template<typename ElementType> InclusiveElementDescendantRange<ElementType>::InclusiveElementDescendantRange(const ContainerNode& root)
    : m_root(root)
{
}

// DoubleElementDescendantRange

template<typename ElementType> DoubleElementDescendantRange<ElementType>::DoubleElementDescendantRange(SingleAdapter&& first, SingleAdapter&& second)
    : m_pair(WTFMove(first), WTFMove(second))
{
}

// DoubleElementDescendantIterator

template<typename ElementType> DoubleElementDescendantIterator<ElementType>::DoubleElementDescendantIterator(SingleIterator&& first, SingleIterator&& second)
    : m_pair(WTFMove(first), WTFMove(second))
{
}

// FilteredElementDescendantIterator

template<typename ElementType, bool filter(const ElementType&)> FilteredElementDescendantIterator<ElementType, filter>::FilteredElementDescendantIterator(const ContainerNode& root, ElementType* element)
    : ElementIterator<ElementType> { &root, element }
{
}

// FilteredElementDescendantRange

template<typename ElementType, bool filter(const ElementType&)> FilteredElementDescendantRange<ElementType, filter>::FilteredElementDescendantRange(const ContainerNode& root)
    : m_root { root }
{
}

// Standalone functions

} // namespace WebCore
