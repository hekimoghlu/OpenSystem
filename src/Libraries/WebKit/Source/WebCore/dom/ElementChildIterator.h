/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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

template<typename> class ElementChildRange;

// Range for iterating through child elements.
template <typename ElementType> ElementChildRange<ElementType> childrenOfType(ContainerNode&);
template <typename ElementType> ElementChildRange<const ElementType> childrenOfType(const ContainerNode&);

template <typename ElementType>
class ElementChildIterator : public ElementIterator<ElementType> {
public:
    ElementChildIterator() = default;
    ElementChildIterator(const ContainerNode& parent, ElementType* current);
    inline ElementChildIterator& operator--();
    inline ElementChildIterator& operator++();
};

template <typename ElementType>
class ElementChildRange {
public:
    ElementChildRange(const ContainerNode& parent);

    inline ElementChildIterator<ElementType> begin() const;
    inline static constexpr std::nullptr_t end() { return nullptr; }
    inline ElementChildIterator<ElementType> beginAt(ElementType&) const;
    
    inline ElementType* first() const;
    inline ElementType* last() const;

private:
    CheckedRef<const ContainerNode> m_parent;
};

// ElementChildIterator

template <typename ElementType>
inline ElementChildIterator<ElementType>::ElementChildIterator(const ContainerNode& parent, ElementType* current)
    : ElementIterator<ElementType>(&parent, current)
{
}

// ElementChildRange

template <typename ElementType>
inline ElementChildRange<ElementType>::ElementChildRange(const ContainerNode& parent)
    : m_parent(parent)
{
}

// Standalone functions

template <typename ElementType>
inline ElementChildRange<ElementType> childrenOfType(ContainerNode& parent)
{
    return ElementChildRange<ElementType>(parent);
}

template <typename ElementType>
inline ElementChildRange<const ElementType> childrenOfType(const ContainerNode& parent)
{
    return ElementChildRange<const ElementType>(parent);
}

} // namespace WebCore
