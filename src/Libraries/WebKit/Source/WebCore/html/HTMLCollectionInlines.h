/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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

#include "HTMLCollection.h"
#include "LiveNodeListInlines.h"

namespace WebCore {

inline ContainerNode& HTMLCollection::rootNode() const
{
    if (isRootedAtTreeScope() && ownerNode().isInTreeScope())
        return ownerNode().treeScope().rootNode();
    return ownerNode();
}

inline const Vector<WeakRef<Element, WeakPtrImplWithEventTargetData>>* CollectionNamedElementCache::findElementsWithId(const AtomString& id) const
{
    return find(m_idMap, id);
}

inline const Vector<WeakRef<Element, WeakPtrImplWithEventTargetData>>* CollectionNamedElementCache::findElementsWithName(const AtomString& name) const
{
    return find(m_nameMap, name);
}

inline void CollectionNamedElementCache::appendToIdCache(const AtomString& id, Element& element)
{
    append(m_idMap, id, element);
}

inline void CollectionNamedElementCache::appendToNameCache(const AtomString& name, Element& element)
{
    append(m_nameMap, name, element);
}

inline void CollectionNamedElementCache::didPopulate()
{
#if ASSERT_ENABLED
    m_didPopulate = true;
#endif
    if (size_t cost = memoryCost())
        reportExtraMemoryAllocatedForCollectionIndexCache(cost);
}

inline const Vector<WeakRef<Element, WeakPtrImplWithEventTargetData>>* CollectionNamedElementCache::find(const StringToElementsMap& map, const AtomString& key) const
{
    ASSERT(m_didPopulate);
    auto it = map.find(key.impl());
    return it != map.end() ? &it->value : nullptr;
}

inline void CollectionNamedElementCache::append(StringToElementsMap& map, const AtomString& key, Element& element)
{
    if (!m_idMap.contains(key.impl()) && !m_nameMap.contains(key.impl()))
        m_propertyNames.append(key);
    map.add(key.impl(), Vector<WeakRef<Element, WeakPtrImplWithEventTargetData>>()).iterator->value.append(element);
}

inline bool HTMLCollection::isRootedAtTreeScope() const
{
    return static_cast<bool>(m_rootType) == static_cast<bool>(RootType::AtTreeScope);
}

inline NodeListInvalidationType HTMLCollection::invalidationType() const
{
    return static_cast<NodeListInvalidationType>(m_invalidationType);
}

inline CollectionType HTMLCollection::type() const
{
    return static_cast<CollectionType>(m_collectionType);
}

inline Document& HTMLCollection::document() const
{
    return m_ownerNode->document();
}

inline void HTMLCollection::invalidateCacheForAttribute(const QualifiedName& attributeName)
{
    if (shouldInvalidateTypeOnAttributeChange(invalidationType(), attributeName))
        invalidateCache();
    else if (hasNamedElementCache() && (attributeName == HTMLNames::idAttr || attributeName == HTMLNames::nameAttr))
        invalidateNamedElementCache(document());
}

inline void HTMLCollection::invalidateCache()
{
    invalidateCacheForDocument(document());
}

inline bool HTMLCollection::hasNamedElementCache() const
{
    return !!m_namedElementCache;
}

inline void HTMLCollection::setNamedItemCache(std::unique_ptr<CollectionNamedElementCache> cache) const
{
    ASSERT(cache);
    ASSERT(!m_namedElementCache);
    cache->didPopulate();
    {
        Locker locker { m_namedElementCacheAssignmentLock };
        m_namedElementCache = WTFMove(cache);
    }
    document().collectionCachedIdNameMap(*this);
}

inline const CollectionNamedElementCache& HTMLCollection::namedItemCaches() const
{
    ASSERT(!!m_namedElementCache);
    return *m_namedElementCache;
}


}
