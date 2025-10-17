/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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

#include "ElementAncestorIterator.h"
#include "ElementIteratorInlines.h"

namespace WebCore {

// ElementAncestorIterator

template <typename ElementType>
inline ElementAncestorIterator<ElementType>& ElementAncestorIterator<ElementType>::operator++()
{
    ElementIterator<ElementType>::traverseAncestor();
    return *this;
}

// ElementAncestorRange

template <typename ElementType>
inline ElementAncestorIterator<ElementType> ElementAncestorRange<ElementType>::begin() const
{
    return ElementAncestorIterator<ElementType>(m_first);
}

// Standalone functions

template<> inline ElementAncestorRange<Element> lineageOfType<Element>(Element& first)
{
    return ElementAncestorRange<Element>(&first);
}

template <typename ElementType>
inline ElementAncestorRange<ElementType> lineageOfType(Element& first)
{
    if (auto* element = dynamicDowncast<ElementType>(first))
        return ElementAncestorRange<ElementType>(element);
    return ancestorsOfType<ElementType>(first);
}

template<> inline ElementAncestorRange<const Element> lineageOfType<Element>(const Element& first)
{
    return ElementAncestorRange<const Element>(&first);
}

template <typename ElementType>
inline ElementAncestorRange<const ElementType> lineageOfType(const Element& first)
{
    if (auto* element = dynamicDowncast<ElementType>(first))
        return ElementAncestorRange<const ElementType>(element);
    return ancestorsOfType<ElementType>(first);
}

template <typename ElementType>
inline ElementAncestorRange<ElementType> lineageOfType(Node& first)
{
    if (auto* element = dynamicDowncast<ElementType>(first))
        return ElementAncestorRange<ElementType>(element);
    return ancestorsOfType<ElementType>(first);
}

template <typename ElementType>
inline ElementAncestorRange<const ElementType> lineageOfType(const Node& first)
{
    if (auto* element = dynamicDowncast<ElementType>(first))
        return ElementAncestorRange<const ElementType>(element);
    return ancestorsOfType<ElementType>(first);
}

template <typename ElementType>
inline ElementAncestorRange<ElementType> ancestorsOfType(Node& descendant)
{
    return ElementAncestorRange<ElementType>(findElementAncestorOfType<ElementType>(descendant));
}

template <typename ElementType>
inline ElementAncestorRange<const ElementType> ancestorsOfType(const Node& descendant)
{
    return ElementAncestorRange<const ElementType>(findElementAncestorOfType<const ElementType>(descendant));
}
}
