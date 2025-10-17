/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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

#include "CachedHTMLCollectionInlines.h"
#include "CollectionIndexCache.h"

namespace WebCore {

template <class Collection, class Iterator>
inline unsigned CollectionIndexCache<Collection, Iterator>::nodeCount(const Collection& collection)
{
    if (!m_nodeCountValid) {
        if (!hasValidCache())
            collection.willValidateIndexCache();
        m_nodeCount = computeNodeCountUpdatingListCache(collection);
        m_nodeCountValid = true;
    }

    return m_nodeCount;
}

template <class Collection, class Iterator>
unsigned CollectionIndexCache<Collection, Iterator>::computeNodeCountUpdatingListCache(const Collection& collection)
{
    auto current = collection.collectionBegin();
    if (!current)
        return 0;

    unsigned oldCapacity = m_cachedList.capacity();
    while (current) {
        m_cachedList.append(*current);
        unsigned traversed;
        collection.collectionTraverseForward(current, 1, traversed);
        ASSERT(traversed == (current ? 1 : 0));
    }
    m_listValid = true;

    if (unsigned capacityDifference = m_cachedList.capacity() - oldCapacity)
        reportExtraMemoryAllocatedForCollectionIndexCache(capacityDifference * sizeof(NodeType*));

    return m_cachedList.size();
}

template <class Collection, class Iterator>
inline typename CollectionIndexCache<Collection, Iterator>::NodeType* CollectionIndexCache<Collection, Iterator>::traverseBackwardTo(const Collection& collection, unsigned index)
{
    ASSERT(m_current);
    ASSERT(index < m_currentIndex);

    bool firstIsCloser = index < m_currentIndex - index;
    if (firstIsCloser || !collection.collectionCanTraverseBackward()) {
        m_current = collection.collectionBegin();
        m_currentIndex = 0;
        if (index)
            collection.collectionTraverseForward(m_current, index, m_currentIndex);
        ASSERT(m_current);
        return &*m_current;
    }

    collection.collectionTraverseBackward(m_current, m_currentIndex - index);
    m_currentIndex = index;

    ASSERT(m_current);
    return &*m_current;
}

template <class Collection, class Iterator>
inline typename CollectionIndexCache<Collection, Iterator>::NodeType* CollectionIndexCache<Collection, Iterator>::traverseForwardTo(const Collection& collection, unsigned index)
{
    ASSERT(m_current);
    ASSERT(index > m_currentIndex);
    ASSERT(!m_nodeCountValid || index < m_nodeCount);

    bool lastIsCloser = m_nodeCountValid && m_nodeCount - index < index - m_currentIndex;
    if (lastIsCloser && collection.collectionCanTraverseBackward()) {
        ASSERT(hasValidCache());
        m_current = collection.collectionLast();
        if (index < m_nodeCount - 1)
            collection.collectionTraverseBackward(m_current, m_nodeCount - index - 1);
        m_currentIndex = index;
        ASSERT(m_current);
        return &*m_current;
    }

    if (!hasValidCache())
        collection.willValidateIndexCache();

    unsigned traversedCount;
    collection.collectionTraverseForward(m_current, index - m_currentIndex, traversedCount);
    m_currentIndex = m_currentIndex + traversedCount;

    if (!m_current) {
        ASSERT(m_currentIndex < index);
        // Failed to find the index but at least we now know the size.
        m_nodeCount = m_currentIndex + 1;
        m_nodeCountValid = true;
        return nullptr;
    }
    ASSERT(hasValidCache());
    return &*m_current;
}

template <class Collection, class Iterator>
inline typename CollectionIndexCache<Collection, Iterator>::NodeType* CollectionIndexCache<Collection, Iterator>::nodeAt(const Collection& collection, unsigned index)
{
    if (m_nodeCountValid && index >= m_nodeCount)
        return nullptr;

    if (m_listValid)
        return m_cachedList[index].get();

    if (m_current) {
        if (index > m_currentIndex)
            return traverseForwardTo(collection, index);
        if (index < m_currentIndex)
            return traverseBackwardTo(collection, index);
        return &*m_current;
    }

    bool lastIsCloser = m_nodeCountValid && m_nodeCount - index < index;
    if (lastIsCloser && collection.collectionCanTraverseBackward()) {
        ASSERT(hasValidCache());
        m_current = collection.collectionLast();
        if (index < m_nodeCount - 1)
            collection.collectionTraverseBackward(m_current, m_nodeCount - index - 1);
        m_currentIndex = index;
        ASSERT(m_current);
        return &*m_current;
    }

    if (!hasValidCache())
        collection.willValidateIndexCache();

    m_current = collection.collectionBegin();
    m_currentIndex = 0;
    bool startIsEnd = !m_current;
    if (index && m_current) {
        collection.collectionTraverseForward(m_current, index, m_currentIndex);
        ASSERT(m_current || m_currentIndex < index);
    }
    if (!m_current) {
        // Failed to find the index but at least we now know the size.
        m_nodeCount = startIsEnd ? 0 : m_currentIndex + 1;
        m_nodeCountValid = true;
        return nullptr;
    }
    ASSERT(hasValidCache());
    return &*m_current;
}

template <class Collection, class Iterator>
void CollectionIndexCache<Collection, Iterator>::invalidate()
{
    m_current = { };
    m_nodeCountValid = false;
    m_listValid = false;
    m_cachedList.shrink(0);
}



}
