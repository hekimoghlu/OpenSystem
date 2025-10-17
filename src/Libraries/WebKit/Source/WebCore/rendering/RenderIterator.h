/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 21, 2024.
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

#include "RenderElement.h"

namespace WebCore {

class RenderText;

template <typename T>
class RenderIterator {
public:
    RenderIterator(const RenderElement* root);
    RenderIterator(const RenderElement* root, T* current);

    T& operator*();
    T* operator->();

    operator bool() const { return m_current; }

    bool operator==(const RenderIterator&) const;

    RenderIterator& traverseNext();
    RenderIterator& traverseNextSibling();
    RenderIterator& traverseNextSkippingChildren();
    RenderIterator& traversePreviousSibling();
    RenderIterator& traverseAncestor();

private:
    const RenderElement* m_root;
    T* m_current;
};

template <typename T>
class RenderConstIterator {
public:
    RenderConstIterator(const RenderElement* root);
    RenderConstIterator(const RenderElement* root, const T* current);

    const T& operator*() const;
    const T* operator->() const;

    operator bool() const { return m_current; }

    bool operator==(const RenderConstIterator& other) const;

    RenderConstIterator& traverseNext();
    RenderConstIterator& traverseNextSibling();
    RenderConstIterator& traverseNextSkippingChildren();
    RenderConstIterator& traversePreviousSibling();
    RenderConstIterator& traverseAncestor();

private:
    const RenderElement* m_root;
    const T* m_current;
};

template <typename T>
class RenderPostOrderIterator {
public:
    RenderPostOrderIterator(const RenderElement* root);
    RenderPostOrderIterator(const RenderElement* root, T* current);

    T& operator*();
    T* operator->();

    operator bool() const { return m_current; }

    bool operator==(const RenderPostOrderIterator&) const;

    RenderPostOrderIterator& traverseNext();

private:
    const RenderElement* m_root;
    T* m_current;
};

template <typename T>
class RenderPostOrderConstIterator {
public:
    RenderPostOrderConstIterator(const RenderElement* root);
    RenderPostOrderConstIterator(const RenderElement* root, const T* current);

    const T& operator*() const;
    const T* operator->() const;

    operator bool() const { return m_current; }

    bool operator==(const RenderPostOrderConstIterator& other) const;

    RenderPostOrderConstIterator& traverseNext();

private:
    const RenderElement* m_root;
    const T* m_current;
};

// Similar to is<>() but without the static_assert() making sure the check is necessary.
template <typename T, typename U>
inline bool isRendererOfType(const U& renderer) { return TypeCastTraits<const T, const U>::isOfType(renderer); }

// Traversal helpers

namespace RenderObjectTraversal {

template <typename U>
inline RenderObject* firstChild(U& object)
{
    return object.firstChild();
}

inline RenderObject* firstChild(RenderObject& object)
{
    return object.firstChildSlow();
}

inline RenderObject* firstChild(RenderText&)
{
    return nullptr;
}

inline RenderObject* nextAncestorSibling(RenderObject& current, const RenderObject* stayWithin)
{
    for (auto* ancestor = current.parent(); ancestor; ancestor = ancestor->parent()) {
        if (ancestor == stayWithin)
            return nullptr;
        if (auto* sibling = ancestor->nextSibling())
            return sibling;
    }
    return nullptr;
}

template <typename U>
inline RenderObject* next(U& current, const RenderObject* stayWithin)
{
    if (auto* child = firstChild(current))
        return child;

    if (&current == stayWithin)
        return nullptr;

    if (auto* sibling = current.nextSibling())
        return sibling;

    return nextAncestorSibling(current, stayWithin);
}

inline RenderObject* nextSkippingChildren(RenderObject& current, const RenderObject* stayWithin)
{
    if (&current == stayWithin)
        return nullptr;

    if (auto* sibling = current.nextSibling())
        return sibling;

    return nextAncestorSibling(current, stayWithin);
}

} // namespace WebCore::RenderObjectTraversal

namespace RenderObjectPostOrderTraversal {

inline RenderObject* next(RenderObject& current, const RenderObject* stayWithin)
{
    if (auto* sibling = current.nextSibling()) {
        if (auto* firstLeafChild = sibling->firstLeafChild())
            return firstLeafChild;
        return sibling;
    }

    auto* parent = current.parent();
    if (parent == stayWithin)
        return nullptr;
    return parent;
}

} // namespace WebCore::RenderObjectPostOrderTraversal

namespace RenderTraversal {

template <typename T, typename U>
inline T* firstChild(U& current)
{
    RenderObject* object = RenderObjectTraversal::firstChild(current);
    while (object && !isRendererOfType<T>(*object))
        object = object->nextSibling();
    return static_cast<T*>(object);
}

template <typename T, typename U>
inline T* lastChild(U& current)
{
    RenderObject* object = current.lastChild();
    while (object && !isRendererOfType<T>(*object))
        object = object->previousSibling();
    return static_cast<T*>(object);
}

template <typename T, typename U>
inline T* nextSibling(U& current)
{
    RenderObject* object = current.nextSibling();
    while (object && !isRendererOfType<T>(*object))
        object = object->nextSibling();
    return static_cast<T*>(object);
}

template <typename T, typename U>
inline T* previousSibling(U& current)
{
    RenderObject* object = current.previousSibling();
    while (object && !isRendererOfType<T>(*object))
        object = object->previousSibling();
    return static_cast<T*>(object);
}

template <typename T>
inline T* findAncestorOfType(const RenderObject& current)
{
    for (auto* ancestor = current.parent(); ancestor; ancestor = ancestor->parent()) {
        if (isRendererOfType<T>(*ancestor))
            return static_cast<T*>(ancestor);
    }
    return nullptr;
}

template <typename T, typename U>
inline T* firstWithin(U& current)
{
    auto* descendant = RenderObjectTraversal::firstChild(current);
    while (descendant && !isRendererOfType<T>(*descendant))
        descendant = RenderObjectTraversal::next(*descendant, &current);
    return static_cast<T*>(descendant);
}

template <typename T, typename U>
inline T* next(U& current, const RenderObject* stayWithin)
{
    auto* descendant = RenderObjectTraversal::next(current, stayWithin);
    while (descendant && !isRendererOfType<T>(*descendant))
        descendant = RenderObjectTraversal::next(*descendant, stayWithin);
    return static_cast<T*>(descendant);
}

} // namespace WebCore::RenderTraversal

namespace RenderPostOrderTraversal {

template <typename T>
inline T* firstWithin(RenderObject& current)
{
    auto* descendant = current.firstLeafChild();
    while (descendant && !isRendererOfType<T>(*descendant))
        descendant = RenderObjectPostOrderTraversal::next(*descendant, &current);
    return static_cast<T*>(descendant);
}

template <typename T>
inline T* next(RenderObject& current, const RenderObject* stayWithin)
{
    auto* descendant = RenderObjectPostOrderTraversal::next(current, stayWithin);
    while (descendant && !isRendererOfType<T>(*descendant))
        descendant = RenderObjectPostOrderTraversal::next(*descendant, stayWithin);
    return static_cast<T*>(descendant);
}

} // namespace WebCore::RenderPostOrderTraversal

// RenderIterator

template <typename T>
inline RenderIterator<T>::RenderIterator(const RenderElement* root)
    : m_root(root)
    , m_current(nullptr)
{
}

template <typename T>
inline RenderIterator<T>::RenderIterator(const RenderElement* root, T* current)
    : m_root(root)
    , m_current(current)
{
}

template <typename T>
inline RenderIterator<T>& RenderIterator<T>::traverseNextSibling()
{
    ASSERT(m_current);
    m_current = RenderTraversal::nextSibling<T>(*m_current);
    return *this;
}

template <typename T>
inline RenderIterator<T>& RenderIterator<T>::traverseNext()
{
    ASSERT(m_current);
    m_current = RenderTraversal::next<T>(*m_current, m_root);
    return *this;
}

template <typename T>
inline RenderIterator<T>& RenderIterator<T>::traverseNextSkippingChildren()
{
    ASSERT(m_current);
    m_current = RenderObjectTraversal::nextSkippingChildren(*m_current, m_root);
    return *this;
}

template <typename T>
inline RenderIterator<T>& RenderIterator<T>::traversePreviousSibling()
{
    ASSERT(m_current);
    m_current = RenderTraversal::previousSibling<T>(*m_current);
    return *this;
}

template <typename T>
inline RenderIterator<T>& RenderIterator<T>::traverseAncestor()
{
    ASSERT(m_current);
    ASSERT(m_current != m_root);
    m_current = RenderTraversal::findAncestorOfType<T>(*m_current);
    return *this;
}

template <typename T>
inline T& RenderIterator<T>::operator*()
{
    ASSERT(m_current);
    return *m_current;
}

template <typename T>
inline T* RenderIterator<T>::operator->()
{
    ASSERT(m_current);
    return m_current;
}

template <typename T>
inline bool RenderIterator<T>::operator==(const RenderIterator& other) const
{
    ASSERT(m_root == other.m_root);
    return m_current == other.m_current;
}

// RenderConstIterator

template <typename T>
inline RenderConstIterator<T>::RenderConstIterator(const RenderElement* root)
    : m_root(root)
    , m_current(nullptr)
{
}

template <typename T>
inline RenderConstIterator<T>::RenderConstIterator(const RenderElement* root, const T* current)
    : m_root(root)
    , m_current(current)
{
}

template <typename T>
inline RenderConstIterator<T>& RenderConstIterator<T>::traverseNextSibling()
{
    ASSERT(m_current);
    m_current = RenderTraversal::nextSibling<T>(*m_current);
    return *this;
}

template <typename T>
inline RenderConstIterator<T>& RenderConstIterator<T>::traverseNext()
{
    ASSERT(m_current);
    m_current = RenderTraversal::next<T>(*m_current, m_root);
    return *this;
}

template <typename T>
inline RenderConstIterator<T>& RenderConstIterator<T>::traverseNextSkippingChildren()
{
    ASSERT(m_current);
    m_current = RenderObjectTraversal::nextSkippingChildren(*m_current, m_root);
    return *this;
}

template <typename T>
inline RenderConstIterator<T>& RenderConstIterator<T>::traversePreviousSibling()
{
    ASSERT(m_current);
    m_current = RenderTraversal::previousSibling<T>(m_current);
    return *this;
}


template <typename T>
inline RenderConstIterator<T>& RenderConstIterator<T>::traverseAncestor()
{
    ASSERT(m_current);
    ASSERT(m_current != m_root);
    m_current = RenderTraversal::findAncestorOfType<const T>(*m_current);
    return *this;
}

template <typename T>
inline const T& RenderConstIterator<T>::operator*() const
{
    ASSERT(m_current);
    return *m_current;
}

template <typename T>
inline const T* RenderConstIterator<T>::operator->() const
{
    ASSERT(m_current);
    return m_current;
}

template <typename T>
inline bool RenderConstIterator<T>::operator==(const RenderConstIterator& other) const
{
    ASSERT(m_root == other.m_root);
    return m_current == other.m_current;
}

// RenderPostOrderIterator

template <typename T>
inline RenderPostOrderIterator<T>::RenderPostOrderIterator(const RenderElement* root)
    : m_root(root)
    , m_current(nullptr)
{
}

template <typename T>
inline RenderPostOrderIterator<T>::RenderPostOrderIterator(const RenderElement* root, T* current)
    : m_root(root)
    , m_current(current)
{
}

template <typename T>
inline RenderPostOrderIterator<T>& RenderPostOrderIterator<T>::traverseNext()
{
    ASSERT(m_current);
    m_current = RenderPostOrderTraversal::next<T>(*m_current, m_root);
    return *this;
}

template <typename T>
inline T& RenderPostOrderIterator<T>::operator*()
{
    ASSERT(m_current);
    return *m_current;
}

template <typename T>
inline T* RenderPostOrderIterator<T>::operator->()
{
    ASSERT(m_current);
    return m_current;
}

template <typename T>
inline bool RenderPostOrderIterator<T>::operator==(const RenderPostOrderIterator& other) const
{
    ASSERT(m_root == other.m_root);
    return m_current == other.m_current;
}

// RenderConstIterator

template <typename T>
inline RenderPostOrderConstIterator<T>::RenderPostOrderConstIterator(const RenderElement* root)
    : m_root(root)
    , m_current(nullptr)
{
}

template <typename T>
inline RenderPostOrderConstIterator<T>::RenderPostOrderConstIterator(const RenderElement* root, const T* current)
    : m_root(root)
    , m_current(current)
{
}

template <typename T>
inline RenderPostOrderConstIterator<T>& RenderPostOrderConstIterator<T>::traverseNext()
{
    ASSERT(m_current);
    m_current = RenderPostOrderTraversal::next<T>(*m_current, m_root);
    return *this;
}

template <typename T>
inline const T& RenderPostOrderConstIterator<T>::operator*() const
{
    ASSERT(m_current);
    return *m_current;
}

template <typename T>
inline const T* RenderPostOrderConstIterator<T>::operator->() const
{
    ASSERT(m_current);
    return m_current;
}

template <typename T>
inline bool RenderPostOrderConstIterator<T>::operator==(const RenderPostOrderConstIterator& other) const
{
    ASSERT(m_root == other.m_root);
    return m_current == other.m_current;
}

} // namespace WebCore

#include "RenderAncestorIterator.h"
#include "RenderChildIterator.h"
