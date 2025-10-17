/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 13, 2024.
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

#include "NodeRareData.h"

namespace WebCore {

inline void NodeListsNodeData::adoptDocument(Document& oldDocument, Document& newDocument)
{
    if (&oldDocument == &newDocument) {
        invalidateCaches();
        return;
    }

    for (auto& cache : m_atomNameCaches.values())
        cache->invalidateCacheForDocument(oldDocument);

    for (auto& list : m_tagCollectionNSCache.values()) {
        ASSERT(!list->isRootedAtTreeScope());
        list->invalidateCacheForDocument(oldDocument);
    }

    for (auto& collection : m_cachedCollections.values())
        collection->invalidateCacheForDocument(oldDocument);
}

void NodeListsNodeData::removeCachedCollection(HTMLCollection* collection, const AtomString& name)
{
    ASSERT(collection == m_cachedCollections.get(namedCollectionKey(collection->type(), name)));
    if (deleteThisAndUpdateNodeRareDataIfAboutToRemoveLastList(collection->protectedOwnerNode()))
        return;
    m_cachedCollections.remove(namedCollectionKey(collection->type(), name));
}

}
