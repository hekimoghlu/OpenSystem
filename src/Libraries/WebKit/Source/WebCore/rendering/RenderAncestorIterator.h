/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 1, 2023.
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
class RenderAncestorIterator : public RenderIterator<T> {
public:
    RenderAncestorIterator();
    explicit RenderAncestorIterator(T* current);
    RenderAncestorIterator& operator++();
};

template <typename T>
class RenderAncestorConstIterator : public RenderConstIterator<T> {
public:
    RenderAncestorConstIterator();
    explicit RenderAncestorConstIterator(const T* current);
    RenderAncestorConstIterator& operator++();
};

template <typename T>
class RenderAncestorIteratorAdapter {
public:
    RenderAncestorIteratorAdapter(T* first);
    RenderAncestorIterator<T> begin();
    RenderAncestorIterator<T> end();
    T* first();

private:
    T* m_first;
};

template <typename T>
class RenderAncestorConstIteratorAdapter {
public:
    RenderAncestorConstIteratorAdapter(const T* first);
    RenderAncestorConstIterator<T> begin() const;
    RenderAncestorConstIterator<T> end() const;
    const T* first() const;

private:
    const T* m_first;
};

template <typename T> RenderAncestorIteratorAdapter<T> ancestorsOfType(RenderObject&);
template <typename T> RenderAncestorConstIteratorAdapter<T> ancestorsOfType(const RenderObject&);
template <typename T> RenderAncestorIteratorAdapter<T> lineageOfType(RenderObject&);
template <typename T> RenderAncestorConstIteratorAdapter<T> lineageOfType(const RenderObject&);

// RenderAncestorIterator

template <typename T>
inline RenderAncestorIterator<T>::RenderAncestorIterator()
    : RenderIterator<T>(nullptr)
{
}

template <typename T>
inline RenderAncestorIterator<T>::RenderAncestorIterator(T* current)
    : RenderIterator<T>(nullptr, current)
{
}

template <typename T>
inline RenderAncestorIterator<T>& RenderAncestorIterator<T>::operator++()
{
    return static_cast<RenderAncestorIterator<T>&>(RenderIterator<T>::traverseAncestor());
}

// RenderAncestorConstIterator

template <typename T>
inline RenderAncestorConstIterator<T>::RenderAncestorConstIterator()
    : RenderConstIterator<T>(nullptr)
{
}

template <typename T>
inline RenderAncestorConstIterator<T>::RenderAncestorConstIterator(const T* current)
    : RenderConstIterator<T>(nullptr, current)
{
}

template <typename T>
inline RenderAncestorConstIterator<T>& RenderAncestorConstIterator<T>::operator++()
{
    return static_cast<RenderAncestorConstIterator<T>&>(RenderConstIterator<T>::traverseAncestor());
}

// RenderAncestorIteratorAdapter

template <typename T>
inline RenderAncestorIteratorAdapter<T>::RenderAncestorIteratorAdapter(T* first)
    : m_first(first)
{
}

template <typename T>
inline RenderAncestorIterator<T> RenderAncestorIteratorAdapter<T>::begin()
{
    return RenderAncestorIterator<T>(m_first);
}

template <typename T>
inline RenderAncestorIterator<T> RenderAncestorIteratorAdapter<T>::end()
{
    return RenderAncestorIterator<T>();
}

template <typename T>
inline T* RenderAncestorIteratorAdapter<T>::first()
{
    return m_first;
}

// RenderAncestorConstIteratorAdapter

template <typename T>
inline RenderAncestorConstIteratorAdapter<T>::RenderAncestorConstIteratorAdapter(const T* first)
    : m_first(first)
{
}

template <typename T>
inline RenderAncestorConstIterator<T> RenderAncestorConstIteratorAdapter<T>::begin() const
{
    return RenderAncestorConstIterator<T>(m_first);
}

template <typename T>
inline RenderAncestorConstIterator<T> RenderAncestorConstIteratorAdapter<T>::end() const
{
    return RenderAncestorConstIterator<T>();
}

template <typename T>
inline const T* RenderAncestorConstIteratorAdapter<T>::first() const
{
    return m_first;
}

// Standalone functions

template <typename T>
inline RenderAncestorIteratorAdapter<T> ancestorsOfType(RenderObject& descendant)
{
    T* first = RenderTraversal::findAncestorOfType<T>(descendant);
    return RenderAncestorIteratorAdapter<T>(first);
}

template <typename T>
inline RenderAncestorConstIteratorAdapter<T> ancestorsOfType(const RenderObject& descendant)
{
    const T* first = RenderTraversal::findAncestorOfType<const T>(descendant);
    return RenderAncestorConstIteratorAdapter<T>(first);
}

template <typename T>
inline RenderAncestorIteratorAdapter<T> lineageOfType(RenderObject& first)
{
    if (isRendererOfType<T>(first))
        return RenderAncestorIteratorAdapter<T>(static_cast<T*>(&first));
    return ancestorsOfType<T>(first);
}

template <typename T>
inline RenderAncestorConstIteratorAdapter<T> lineageOfType(const RenderObject& first)
{
    if (isRendererOfType<T>(first))
        return RenderAncestorConstIteratorAdapter<T>(static_cast<const T*>(&first));
    return ancestorsOfType<T>(first);
}

} // namespace WebCore
