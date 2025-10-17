/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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

template<typename> class ElementAncestorRange;

// Range for iterating an element and its ancestors.
template<typename ElementType>
inline ElementAncestorRange<ElementType> lineageOfType(Element& first);
template<typename ElementType>
inline ElementAncestorRange<const ElementType> lineageOfType(const Element& first);

// Range for iterating a node's element ancestors.
template<typename ElementType>
inline ElementAncestorRange<ElementType> ancestorsOfType(Node& descendant);
template<typename ElementType>
inline ElementAncestorRange<const ElementType> ancestorsOfType(const Node& descendant);

template <typename ElementType>
class ElementAncestorIterator : public ElementIterator<ElementType> {
public:
    explicit ElementAncestorIterator(ElementType* = nullptr);
    inline ElementAncestorIterator& operator++();
};

template <typename ElementType>
class ElementAncestorRange {
public:
    explicit ElementAncestorRange(ElementType* first);
    ElementAncestorIterator<ElementType> begin() const;
    static constexpr std::nullptr_t end() { return nullptr; }
    ElementType* first() const { return m_first; }

private:
    ElementType* const m_first;
};

// ElementAncestorIterator

template <typename ElementType>
inline ElementAncestorIterator<ElementType>::ElementAncestorIterator(ElementType* current)
    : ElementIterator<ElementType>(nullptr, current)
{
}

// ElementAncestorRange

template <typename ElementType>
inline ElementAncestorRange<ElementType>::ElementAncestorRange(ElementType* first)
    : m_first(first)
{
}

} // namespace WebCore
