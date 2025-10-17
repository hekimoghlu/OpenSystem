/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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

#include <wtf/Vector.h>

namespace WebCore {

class WeakPtrImplWithEventTargetData;

WEBCORE_EXPORT void reportExtraMemoryAllocatedForCollectionIndexCache(size_t);

template <class Collection, class Iterator>
class CollectionIndexCache {
public:
    CollectionIndexCache();

    typedef typename std::iterator_traits<Iterator>::value_type NodeType;

    inline unsigned nodeCount(const Collection&);
    inline NodeType* nodeAt(const Collection&, unsigned index);

    bool hasValidCache() const { return m_current || m_nodeCountValid || m_listValid; }
    inline void invalidate();
    size_t memoryCost()
    {
        // memoryCost() may be invoked concurrently from a GC thread, and we need to be careful
        // about what data we access here and how. Accessing m_cachedList.capacity() is safe
        // because it doesn't involve any pointer chasing.
        return m_cachedList.capacity() * sizeof(NodeType*);
    }

private:
    inline unsigned computeNodeCountUpdatingListCache(const Collection&);
    inline NodeType* traverseBackwardTo(const Collection&, unsigned);
    inline NodeType* traverseForwardTo(const Collection&, unsigned);

    Iterator m_current { };
    unsigned m_currentIndex { 0 };
    unsigned m_nodeCount { 0 };
    Vector<WeakPtr<NodeType, WeakPtrImplWithEventTargetData>> m_cachedList;
    bool m_nodeCountValid : 1;
    bool m_listValid : 1;
};

template<class Collection, class Iterator> CollectionIndexCache<Collection, Iterator>::CollectionIndexCache()
    : m_nodeCountValid(false)
    , m_listValid(false)
{
}

}
