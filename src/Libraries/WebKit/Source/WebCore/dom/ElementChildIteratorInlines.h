/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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

#include "ElementChildIterator.h"
#include "ElementIteratorInlines.h"

namespace WebCore {

// ElementChildIterator

template <typename ElementType>
inline ElementChildIterator<ElementType>& ElementChildIterator<ElementType>::operator--()
{
    ElementIterator<ElementType>::traversePreviousSibling();
    return *this;
}

template <typename ElementType>
inline ElementChildIterator<ElementType>& ElementChildIterator<ElementType>::operator++()
{
    ElementIterator<ElementType>::traverseNextSibling();
    return *this;
}

// ElementChildRange

template <typename ElementType>
inline ElementChildIterator<ElementType> ElementChildRange<ElementType>::begin() const
{
    return ElementChildIterator<ElementType>(m_parent, Traversal<ElementType>::firstChild(m_parent));
}

template <typename ElementType>
inline ElementType* ElementChildRange<ElementType>::first() const
{
    return Traversal<ElementType>::firstChild(m_parent);
}

template <typename ElementType>
inline ElementType* ElementChildRange<ElementType>::last() const
{
    return Traversal<ElementType>::lastChild(m_parent);
}

template <typename ElementType>
inline ElementChildIterator<ElementType> ElementChildRange<ElementType>::beginAt(ElementType& child) const
{
    ASSERT(child.parentNode() == m_parent.ptr());
    return ElementChildIterator<ElementType>(m_parent.get(), &child);
}

}
