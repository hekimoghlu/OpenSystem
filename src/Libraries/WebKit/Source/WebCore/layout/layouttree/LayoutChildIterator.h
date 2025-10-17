/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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
class LayoutChildIterator : public LayoutIterator<T> {
public:
    LayoutChildIterator(const ElementBox& parent);
    LayoutChildIterator(const ElementBox& parent, const T* current);
    LayoutChildIterator& operator++();
};

template <typename T>
class LayoutChildIteratorAdapter {
public:
    LayoutChildIteratorAdapter(const ElementBox& parent);
    LayoutChildIterator<T> begin() const;
    LayoutChildIterator<T> end() const;
    const T* first() const;
    const T* last() const;

private:
    const ElementBox& m_parent;
};

template <typename T> LayoutChildIteratorAdapter<T> childrenOfType(const ElementBox&);

// LayoutChildIterator

template <typename T>
inline LayoutChildIterator<T>::LayoutChildIterator(const ElementBox& parent)
    : LayoutIterator<T>(&parent)
{
}

template <typename T>
inline LayoutChildIterator<T>::LayoutChildIterator(const ElementBox& parent, const T* current)
    : LayoutIterator<T>(&parent, current)
{
}

template <typename T>
inline LayoutChildIterator<T>& LayoutChildIterator<T>::operator++()
{
    return static_cast<LayoutChildIterator<T>&>(LayoutIterator<T>::traverseNextSibling());
}

// LayoutChildIteratorAdapter

template <typename T>
inline LayoutChildIteratorAdapter<T>::LayoutChildIteratorAdapter(const ElementBox& parent)
    : m_parent(parent)
{
}

template <typename T>
inline LayoutChildIterator<T> LayoutChildIteratorAdapter<T>::begin() const
{
    return LayoutChildIterator<T>(m_parent, Traversal::firstChild<T>(m_parent));
}

template <typename T>
inline LayoutChildIterator<T> LayoutChildIteratorAdapter<T>::end() const
{
    return LayoutChildIterator<T>(m_parent);
}

template <typename T>
inline const T* LayoutChildIteratorAdapter<T>::first() const
{
    return Traversal::firstChild<T>(m_parent);
}

template <typename T>
inline LayoutChildIteratorAdapter<T> childrenOfType(const ElementBox& parent)
{
    return LayoutChildIteratorAdapter<T>(parent);
}

}
}
