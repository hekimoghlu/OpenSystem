/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 17, 2022.
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

#include "LayoutIterator.h"

namespace WebCore {
namespace Layout {

template <typename T>
class LayoutDescendantIterator : public LayoutIterator<T> {
public:
    LayoutDescendantIterator(const ElementBox& root);
    LayoutDescendantIterator(const ElementBox& root, const T* current);
    LayoutDescendantIterator& operator++();
};

template <typename T>
class LayoutDescendantIteratorAdapter {
public:
    LayoutDescendantIteratorAdapter(const ElementBox& root);
    LayoutDescendantIterator<T> begin();
    LayoutDescendantIterator<T> end();
    LayoutDescendantIterator<T> at(const T&);

private:
    const ElementBox& m_root;
};

template <typename T> LayoutDescendantIteratorAdapter<T> descendantsOfType(const Box&);

// LayoutDescendantIterator

template <typename T>
inline LayoutDescendantIterator<T>::LayoutDescendantIterator(const ElementBox& root)
    : LayoutIterator<T>(&root)
{
}

template <typename T>
inline LayoutDescendantIterator<T>::LayoutDescendantIterator(const ElementBox& root, const T* current)
    : LayoutIterator<T>(&root, current)
{
}

template <typename T>
inline LayoutDescendantIterator<T>& LayoutDescendantIterator<T>::operator++()
{
    return static_cast<LayoutDescendantIterator<T>&>(LayoutIterator<T>::traverseNext());
}

// LayoutDescendantIteratorAdapter

template <typename T>
inline LayoutDescendantIteratorAdapter<T>::LayoutDescendantIteratorAdapter(const ElementBox& root)
    : m_root(root)
{
}

template <typename T>
inline LayoutDescendantIterator<T> LayoutDescendantIteratorAdapter<T>::begin()
{
    return LayoutDescendantIterator<T>(m_root, Traversal::firstWithin<T>(m_root));
}

template <typename T>
inline LayoutDescendantIterator<T> LayoutDescendantIteratorAdapter<T>::end()
{
    return LayoutDescendantIterator<T>(m_root);
}

template <typename T>
inline LayoutDescendantIterator<T> LayoutDescendantIteratorAdapter<T>::at(const T& current)
{
    return LayoutDescendantIterator<T>(m_root, &current);
}

// Standalone functions

template <typename T>
inline LayoutDescendantIteratorAdapter<T> descendantsOfType(const ElementBox& root)
{
    return LayoutDescendantIteratorAdapter<T>(root);
}

}
}
