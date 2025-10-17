/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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

#include "RenderIterator.h"

namespace WebCore {

template <typename T>
class RenderChildIterator : public RenderIterator<T> {
public:
    RenderChildIterator(const RenderElement& parent);
    RenderChildIterator(const RenderElement& parent, T* current);
    RenderChildIterator& operator++();
};

template <typename T>
class RenderChildConstIterator : public RenderConstIterator<T> {
public:
    RenderChildConstIterator(const RenderElement& parent);
    RenderChildConstIterator(const RenderElement& parent, const T* current);
    RenderChildConstIterator& operator++();
};

template <typename T>
class RenderChildIteratorAdapter {
public:
    RenderChildIteratorAdapter(RenderElement& parent);
    RenderChildIterator<T> begin();
    RenderChildIterator<T> end();
    T* first();
    T* last();

private:
    CheckedRef<RenderElement> m_parent;
};

template <typename T>
class RenderChildConstIteratorAdapter {
public:
    RenderChildConstIteratorAdapter(const RenderElement& parent);
    RenderChildConstIterator<T> begin() const;
    RenderChildConstIterator<T> end() const;
    const T* first() const;
    const T* last() const;

private:
    CheckedRef<const RenderElement> m_parent;
};

template <typename T> RenderChildIteratorAdapter<T> childrenOfType(RenderElement&);
template <typename T> RenderChildConstIteratorAdapter<T> childrenOfType(const RenderElement&);

// RenderChildIterator

template <typename T>
inline RenderChildIterator<T>::RenderChildIterator(const RenderElement& parent)
    : RenderIterator<T>(&parent)
{
}

template <typename T>
inline RenderChildIterator<T>::RenderChildIterator(const RenderElement& parent, T* current)
    : RenderIterator<T>(&parent, current)
{
}

template <typename T>
inline RenderChildIterator<T>& RenderChildIterator<T>::operator++()
{
    return static_cast<RenderChildIterator<T>&>(RenderIterator<T>::traverseNextSibling());
}

// RenderChildConstIterator

template <typename T>
inline RenderChildConstIterator<T>::RenderChildConstIterator(const RenderElement& parent)
    : RenderConstIterator<T>(&parent)
{
}

template <typename T>
inline RenderChildConstIterator<T>::RenderChildConstIterator(const RenderElement& parent, const T* current)
    : RenderConstIterator<T>(&parent, current)
{
}

template <typename T>
inline RenderChildConstIterator<T>& RenderChildConstIterator<T>::operator++()
{
    return static_cast<RenderChildConstIterator<T>&>(RenderConstIterator<T>::traverseNextSibling());
}

// RenderChildIteratorAdapter

template <typename T>
inline RenderChildIteratorAdapter<T>::RenderChildIteratorAdapter(RenderElement& parent)
    : m_parent(parent)
{
}

template <typename T>
inline RenderChildIterator<T> RenderChildIteratorAdapter<T>::begin()
{
    return RenderChildIterator<T>(m_parent, RenderTraversal::firstChild<T>(m_parent.get()));
}

template <typename T>
inline RenderChildIterator<T> RenderChildIteratorAdapter<T>::end()
{
    return RenderChildIterator<T>(m_parent.get());
}

template <typename T>
inline T* RenderChildIteratorAdapter<T>::first()
{
    return RenderTraversal::firstChild<T>(m_parent.get());
}

template <typename T>
inline T* RenderChildIteratorAdapter<T>::last()
{
    return RenderTraversal::lastChild<T>(m_parent.get());
}

// RenderChildConstIteratorAdapter

template <typename T>
inline RenderChildConstIteratorAdapter<T>::RenderChildConstIteratorAdapter(const RenderElement& parent)
    : m_parent(parent)
{
}

template <typename T>
inline RenderChildConstIterator<T> RenderChildConstIteratorAdapter<T>::begin() const
{
    return RenderChildConstIterator<T>(m_parent.get(), RenderTraversal::firstChild<T>(m_parent.get()));
}

template <typename T>
inline RenderChildConstIterator<T> RenderChildConstIteratorAdapter<T>::end() const
{
    return RenderChildConstIterator<T>(m_parent.get());
}

template <typename T>
inline const T* RenderChildConstIteratorAdapter<T>::first() const
{
    return RenderTraversal::firstChild<T>(m_parent.get());
}

template <typename T>
inline const T* RenderChildConstIteratorAdapter<T>::last() const
{
    return RenderTraversal::lastChild<T>(m_parent.get());
}

// Standalone functions

template <typename T>
inline RenderChildIteratorAdapter<T> childrenOfType(RenderElement& parent)
{
    return RenderChildIteratorAdapter<T>(parent);
}

template <typename T>
inline RenderChildConstIteratorAdapter<T> childrenOfType(const RenderElement& parent)
{
    return RenderChildConstIteratorAdapter<T>(parent);
}

} // namespace WebCore
