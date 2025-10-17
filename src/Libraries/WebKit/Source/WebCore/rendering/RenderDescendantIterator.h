/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 3, 2024.
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
class RenderDescendantIterator : public RenderIterator<T> {
public:
    RenderDescendantIterator(const RenderElement& root);
    RenderDescendantIterator(const RenderElement& root, T* current);
    RenderDescendantIterator& operator++();
};

template <typename T>
class RenderDescendantConstIterator : public RenderConstIterator<T> {
public:
    RenderDescendantConstIterator(const RenderElement& root);
    RenderDescendantConstIterator(const RenderElement& root, const T* current);
    RenderDescendantConstIterator& operator++();
};

template <typename T>
class RenderDescendantIteratorAdapter {
public:
    RenderDescendantIteratorAdapter(RenderElement& root);
    RenderDescendantIterator<T> begin();
    RenderDescendantIterator<T> end();
    RenderDescendantIterator<T> at(T&);

private:
    RenderElement& m_root;
};

template <typename T>
class RenderDescendantConstIteratorAdapter {
public:
    RenderDescendantConstIteratorAdapter(const RenderElement& root);
    RenderDescendantConstIterator<T> begin() const;
    RenderDescendantConstIterator<T> end() const;
    RenderDescendantConstIterator<T> at(const T&) const;

private:
    const RenderElement& m_root;
};

template <typename T>
class RenderDescendantPostOrderIterator : public RenderPostOrderIterator<T> {
public:
    RenderDescendantPostOrderIterator(const RenderElement& root);
    RenderDescendantPostOrderIterator(const RenderElement& root, T* current);
    RenderDescendantPostOrderIterator& operator++();
};

template <typename T>
class RenderDescendantPostOrderConstIterator : public RenderPostOrderConstIterator<T> {
public:
    RenderDescendantPostOrderConstIterator(const RenderElement& root);
    RenderDescendantPostOrderConstIterator(const RenderElement& root, const T* current);
    RenderDescendantPostOrderConstIterator& operator++();
};

template <typename T>
class RenderDescendantPostOrderIteratorAdapter {
public:
    RenderDescendantPostOrderIteratorAdapter(RenderElement& root);
    RenderDescendantPostOrderIterator<T> begin();
    RenderDescendantPostOrderIterator<T> end();
    RenderDescendantPostOrderIterator<T> at(T&);

private:
    RenderElement& m_root;
};

template <typename T>
class RenderDescendantPostOrderConstIteratorAdapter {
public:
    RenderDescendantPostOrderConstIteratorAdapter(const RenderElement& root);
    RenderDescendantPostOrderConstIterator<T> begin() const;
    RenderDescendantPostOrderConstIterator<T> end() const;
    RenderDescendantPostOrderConstIterator<T> at(const T&) const;

private:
    const RenderElement& m_root;
};

template <typename T> RenderDescendantIteratorAdapter<T> descendantsOfType(RenderElement&);
template <typename T> RenderDescendantConstIteratorAdapter<T> descendantsOfType(const RenderElement&);

template <typename T> RenderDescendantPostOrderIteratorAdapter<T> descendantsOfTypePostOrder(RenderElement&);
template <typename T> RenderDescendantPostOrderConstIteratorAdapter<T> descendantsOfTypePostOrder(const RenderElement&);

// RenderDescendantIterator

template <typename T>
inline RenderDescendantIterator<T>::RenderDescendantIterator(const RenderElement& root)
    : RenderIterator<T>(&root)
{
}

template <typename T>
inline RenderDescendantIterator<T>::RenderDescendantIterator(const RenderElement& root, T* current)
    : RenderIterator<T>(&root, current)
{
}

template <typename T>
inline RenderDescendantIterator<T>& RenderDescendantIterator<T>::operator++()
{
    return static_cast<RenderDescendantIterator<T>&>(RenderIterator<T>::traverseNext());
}

// RenderDescendantConstIterator

template <typename T>
inline RenderDescendantConstIterator<T>::RenderDescendantConstIterator(const RenderElement& root)
    : RenderConstIterator<T>(&root)
{
}

template <typename T>
inline RenderDescendantConstIterator<T>::RenderDescendantConstIterator(const RenderElement& root, const T* current)
    : RenderConstIterator<T>(&root, current)
{
}

template <typename T>
inline RenderDescendantConstIterator<T>& RenderDescendantConstIterator<T>::operator++()
{
    return static_cast<RenderDescendantConstIterator<T>&>(RenderConstIterator<T>::traverseNext());
}

// RenderDescendantIteratorAdapter

template <typename T>
inline RenderDescendantIteratorAdapter<T>::RenderDescendantIteratorAdapter(RenderElement& root)
    : m_root(root)
{
}

template <typename T>
inline RenderDescendantIterator<T> RenderDescendantIteratorAdapter<T>::begin()
{
    return RenderDescendantIterator<T>(m_root, RenderTraversal::firstWithin<T>(m_root));
}

template <typename T>
inline RenderDescendantIterator<T> RenderDescendantIteratorAdapter<T>::end()
{
    return RenderDescendantIterator<T>(m_root);
}

template <typename T>
inline RenderDescendantIterator<T> RenderDescendantIteratorAdapter<T>::at(T& current)
{
    return RenderDescendantIterator<T>(m_root, &current);
}

// RenderDescendantConstIteratorAdapter

template <typename T>
inline RenderDescendantConstIteratorAdapter<T>::RenderDescendantConstIteratorAdapter(const RenderElement& root)
    : m_root(root)
{
}

template <typename T>
inline RenderDescendantConstIterator<T> RenderDescendantConstIteratorAdapter<T>::begin() const
{
    return RenderDescendantConstIterator<T>(m_root, RenderTraversal::firstWithin<T>(m_root));
}

template <typename T>
inline RenderDescendantConstIterator<T> RenderDescendantConstIteratorAdapter<T>::end() const
{
    return RenderDescendantConstIterator<T>(m_root);
}

template <typename T>
inline RenderDescendantConstIterator<T> RenderDescendantConstIteratorAdapter<T>::at(const T& current) const
{
    return RenderDescendantConstIterator<T>(m_root, &current);
}

// RenderDescendantPostOrderIterator

template <typename T>
inline RenderDescendantPostOrderIterator<T>::RenderDescendantPostOrderIterator(const RenderElement& root)
    : RenderPostOrderIterator<T>(&root)
{
}

template <typename T>
inline RenderDescendantPostOrderIterator<T>::RenderDescendantPostOrderIterator(const RenderElement& root, T* current)
    : RenderPostOrderIterator<T>(&root, current)
{
}

template <typename T>
inline RenderDescendantPostOrderIterator<T>& RenderDescendantPostOrderIterator<T>::operator++()
{
    return static_cast<RenderDescendantPostOrderIterator<T>&>(RenderPostOrderIterator<T>::traverseNext());
}

// RenderDescendantPostOrderConstIterator

template <typename T>
inline RenderDescendantPostOrderConstIterator<T>::RenderDescendantPostOrderConstIterator(const RenderElement& root)
    : RenderPostOrderConstIterator<T>(&root)
{
}

template <typename T>
inline RenderDescendantPostOrderConstIterator<T>::RenderDescendantPostOrderConstIterator(const RenderElement& root, const T* current)
    : RenderPostOrderConstIterator<T>(&root, current)
{
}

template <typename T>
inline RenderDescendantPostOrderConstIterator<T>& RenderDescendantPostOrderConstIterator<T>::operator++()
{
    return static_cast<RenderDescendantPostOrderConstIterator<T>&>(RenderPostOrderConstIterator<T>::traverseNext());
}

// RenderDescendantPostOrderIteratorAdapter

template <typename T>
inline RenderDescendantPostOrderIteratorAdapter<T>::RenderDescendantPostOrderIteratorAdapter(RenderElement& root)
    : m_root(root)
{
}

template <typename T>
inline RenderDescendantPostOrderIterator<T> RenderDescendantPostOrderIteratorAdapter<T>::begin()
{
    return RenderDescendantPostOrderIterator<T>(m_root, RenderPostOrderTraversal::firstWithin<T>(m_root));
}

template <typename T>
inline RenderDescendantPostOrderIterator<T> RenderDescendantPostOrderIteratorAdapter<T>::end()
{
    return RenderDescendantPostOrderIterator<T>(m_root);
}

template <typename T>
inline RenderDescendantPostOrderIterator<T> RenderDescendantPostOrderIteratorAdapter<T>::at(T& current)
{
    return RenderDescendantPostOrderIterator<T>(m_root, &current);
}

// RenderDescendantPostOrderConstIteratorAdapter

template <typename T>
inline RenderDescendantPostOrderConstIteratorAdapter<T>::RenderDescendantPostOrderConstIteratorAdapter(const RenderElement& root)
    : m_root(root)
{
}

template <typename T>
inline RenderDescendantPostOrderConstIterator<T> RenderDescendantPostOrderConstIteratorAdapter<T>::begin() const
{
    return RenderDescendantPostOrderConstIterator<T>(m_root, RenderPostOrderTraversal::firstWithin<T>(m_root));
}

template <typename T>
inline RenderDescendantPostOrderConstIterator<T> RenderDescendantPostOrderConstIteratorAdapter<T>::end() const
{
    return RenderDescendantPostOrderConstIterator<T>(m_root);
}

template <typename T>
inline RenderDescendantPostOrderConstIterator<T> RenderDescendantPostOrderConstIteratorAdapter<T>::at(const T& current) const
{
    return RenderDescendantPostOrderConstIterator<T>(m_root, &current);
}

// Standalone functions

template <typename T>
inline RenderDescendantIteratorAdapter<T> descendantsOfType(RenderElement& root)
{
    return RenderDescendantIteratorAdapter<T>(root);
}

template <typename T>
inline RenderDescendantConstIteratorAdapter<T> descendantsOfType(const RenderElement& root)
{
    return RenderDescendantConstIteratorAdapter<T>(root);
}

template <typename T>
inline RenderDescendantPostOrderIteratorAdapter<T> descendantsOfTypePostOrder(RenderElement& root)
{
    return RenderDescendantPostOrderIteratorAdapter<T>(root);
}

template <typename T>
inline RenderDescendantPostOrderConstIteratorAdapter<T> descendantsOfTypePostOrder(const RenderElement& root)
{
    return RenderDescendantPostOrderConstIteratorAdapter<T>(root);
}

} // namespace WebCore
