/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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

#include "ElementInlines.h"
#include "ElementIterator.h"
#include "ElementTraversal.h"

namespace WebCore {

template <typename ElementType>
inline ElementIterator<ElementType>& ElementIterator<ElementType>::traverseNext()
{
    ASSERT(m_current);
    ASSERT(!m_assertions.domTreeHasMutated());
    m_current = Traversal<ElementType>::next(*m_current, m_root.get());
#if ASSERT_ENABLED
    // Drop the assertion when the iterator reaches the end.
    if (!m_current)
        m_assertions.dropEventDispatchAssertion();
#endif
    return *this;
}

template <typename ElementType>
inline ElementIterator<ElementType>& ElementIterator<ElementType>::traversePrevious()
{
    ASSERT(m_current);
    ASSERT(!m_assertions.domTreeHasMutated());
    m_current = Traversal<ElementType>::previous(*m_current, m_root.get());
#if ASSERT_ENABLED
    // Drop the assertion when the iterator reaches the end.
    if (!m_current)
        m_assertions.dropEventDispatchAssertion();
#endif
    return *this;
}

template <typename ElementType>
inline ElementIterator<ElementType>& ElementIterator<ElementType>::traverseNextSibling()
{
    ASSERT(m_current);
    ASSERT(!m_assertions.domTreeHasMutated());
    m_current = Traversal<ElementType>::nextSibling(*m_current);
#if ASSERT_ENABLED
    // Drop the assertion when the iterator reaches the end.
    if (!m_current)
        m_assertions.dropEventDispatchAssertion();
#endif
    return *this;
}

template <typename ElementType>
inline ElementIterator<ElementType>& ElementIterator<ElementType>::traversePreviousSibling()
{
    ASSERT(m_current);
    ASSERT(!m_assertions.domTreeHasMutated());
    m_current = Traversal<ElementType>::previousSibling(*m_current);
#if ASSERT_ENABLED
    // Drop the assertion when the iterator reaches the end.
    if (!m_current)
        m_assertions.dropEventDispatchAssertion();
#endif
    return *this;
}

template <typename ElementType>
inline ElementIterator<ElementType>& ElementIterator<ElementType>::traverseNextSkippingChildren()
{
    ASSERT(m_current);
    ASSERT(!m_assertions.domTreeHasMutated());
    m_current = Traversal<ElementType>::nextSkippingChildren(*m_current, m_root.get());
#if ASSERT_ENABLED
    // Drop the assertion when the iterator reaches the end.
    if (!m_current)
        m_assertions.dropEventDispatchAssertion();
#endif
    return *this;
}

template <typename ElementType>
inline ElementType* findElementAncestorOfType(const Node& current)
{
    for (Element* ancestor = current.parentElement(); ancestor; ancestor = ancestor->parentElement()) {
        if (auto* element = dynamicDowncast<ElementType>(*ancestor))
            return element;
    }
    return nullptr;
}

template <>
inline Element* findElementAncestorOfType<Element>(const Node& current)
{
    return current.parentElement();
}

template <typename ElementType>
inline ElementIterator<ElementType>& ElementIterator<ElementType>::traverseAncestor()
{
    ASSERT(m_current);
    ASSERT(m_current != m_root);
    ASSERT(!m_assertions.domTreeHasMutated());
    m_current = findElementAncestorOfType<ElementType>(*m_current);
#if ASSERT_ENABLED
    // Drop the assertion when the iterator reaches the end.
    if (!m_current)
        m_assertions.dropEventDispatchAssertion();
#endif
    return *this;
}

}
