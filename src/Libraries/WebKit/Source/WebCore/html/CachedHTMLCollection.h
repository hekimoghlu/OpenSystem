/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 15, 2023.
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

#include "CollectionTraversal.h"
#include "HTMLCollection.h"
#include "HTMLElement.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

template <typename HTMLCollectionClass, CollectionTraversalType traversalType>
class CachedHTMLCollection : public HTMLCollection {
    WTF_MAKE_TZONE_OR_ISO_NON_HEAP_ALLOCATABLE(CachedHTMLCollection);
public:
    CachedHTMLCollection(ContainerNode& base, CollectionType);
    
    virtual ~CachedHTMLCollection();
    
    unsigned length() const override;
    Element* item(unsigned offset) const override;
    Element* namedItem(const AtomString&) const override;
    size_t memoryCost() const final;

    // For CollectionIndexCache; do not use elsewhere.
    using Traversal = CollectionTraversal<traversalType>;
    using Iterator = typename Traversal::Iterator;
    inline Iterator collectionBegin() const;
    inline Iterator collectionLast() const;
    inline void collectionTraverseForward(Iterator& current, unsigned count, unsigned& traversedCount) const;
    inline void collectionTraverseBackward(Iterator& current, unsigned count) const;
    inline bool collectionCanTraverseBackward() const;
    void willValidateIndexCache() const { document().registerCollection(const_cast<CachedHTMLCollection&>(*this)); }

    void invalidateCacheForDocument(Document&) override;
    
    inline bool elementMatches(Element&) const;
    
private:
    HTMLCollectionClass& collection() { return static_cast<HTMLCollectionClass&>(*this); }
    const HTMLCollectionClass& collection() const { return static_cast<const HTMLCollectionClass&>(*this); }
    
    mutable CollectionIndexCache<HTMLCollectionClass, Iterator> m_indexCache;
};

template <typename HTMLCollectionClass, CollectionTraversalType traversalType>
CachedHTMLCollection<HTMLCollectionClass, traversalType>::CachedHTMLCollection(ContainerNode& base, CollectionType collectionType)
    : HTMLCollection(base, collectionType)
{
}

template <typename HTMLCollectionClass, CollectionTraversalType traversalType>
bool CachedHTMLCollection<HTMLCollectionClass, traversalType>::elementMatches(Element&) const
{
    // We call the elementMatches() method directly on the subclass instead for performance.
    ASSERT_NOT_REACHED();
    return false;
}

} // namespace WebCore
