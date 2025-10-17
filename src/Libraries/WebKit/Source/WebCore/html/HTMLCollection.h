/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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

#include "CollectionIndexCache.h"
#include "Element.h"
#include "HTMLNames.h"
#include "LiveNodeList.h"
#include <wtf/HashMap.h>

namespace WebCore {

class CollectionNamedElementCache {
    WTF_MAKE_TZONE_ALLOCATED(CollectionNamedElementCache);
public:
    inline const Vector<WeakRef<Element, WeakPtrImplWithEventTargetData>>* findElementsWithId(const AtomString& id) const;
    inline const Vector<WeakRef<Element, WeakPtrImplWithEventTargetData>>* findElementsWithName(const AtomString& name) const;
    const Vector<AtomString>& propertyNames() const { return m_propertyNames; }
    
    inline void appendToIdCache(const AtomString& id, Element&);
    inline void appendToNameCache(const AtomString& name, Element&);
    inline void didPopulate();

    inline size_t memoryCost() const;

private:
    typedef UncheckedKeyHashMap<AtomStringImpl*, Vector<WeakRef<Element, WeakPtrImplWithEventTargetData>>> StringToElementsMap;

    inline const Vector<WeakRef<Element, WeakPtrImplWithEventTargetData>>* find(const StringToElementsMap&, const AtomString& key) const;
    inline void append(StringToElementsMap&, const AtomString& key, Element&);

    StringToElementsMap m_idMap;
    StringToElementsMap m_nameMap;
    Vector<AtomString> m_propertyNames;

#if ASSERT_ENABLED
    bool m_didPopulate { false };
#endif
};

// HTMLCollection subclasses NodeList to maintain legacy ObjC API compatibility.
class HTMLCollection : public NodeList {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(HTMLCollection, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT virtual ~HTMLCollection();

    // DOM API
    Element* item(unsigned index) const override = 0; // Tighten return type from NodeList::item().
    virtual Element* namedItem(const AtomString& name) const = 0;
    const Vector<AtomString>& supportedPropertyNames();
    bool isSupportedPropertyName(const AtomString& name);

    // Non-DOM API
    Vector<Ref<Element>> namedItems(const AtomString& name) const;
    size_t memoryCost() const override;

    inline bool isRootedAtTreeScope() const;
    inline NodeListInvalidationType invalidationType() const;
    inline CollectionType type() const;
    inline ContainerNode& ownerNode() const;
    inline Ref<ContainerNode> protectedOwnerNode() const;
    inline ContainerNode& rootNode() const;
    inline void invalidateCacheForAttribute(const QualifiedName& attributeName);
    WEBCORE_EXPORT virtual void invalidateCacheForDocument(Document&);
    inline void invalidateCache();

    inline bool hasNamedElementCache() const;

protected:
    HTMLCollection(ContainerNode& base, CollectionType);

    WEBCORE_EXPORT virtual void updateNamedElementCache() const;
    WEBCORE_EXPORT Element* namedItemSlow(const AtomString& name) const;

    inline void setNamedItemCache(std::unique_ptr<CollectionNamedElementCache>) const;
    inline const CollectionNamedElementCache& namedItemCaches() const;

    inline Document& document() const;

    void invalidateNamedElementCache(Document&) const;

    enum class RootType : bool { AtNode, AtTreeScope };
    static RootType rootTypeFromCollectionType(CollectionType);

    mutable Lock m_namedElementCacheAssignmentLock;

    const unsigned m_collectionType : 5; // CollectionType
    const unsigned m_invalidationType : 4; // NodeListInvalidationType
    const unsigned m_rootType : 1; // RootType

    Ref<ContainerNode> m_ownerNode;

    mutable std::unique_ptr<CollectionNamedElementCache> m_namedElementCache;
};

inline size_t CollectionNamedElementCache::memoryCost() const
{
    // memoryCost() may be invoked concurrently from a GC thread, and we need to be careful about what data we access here and how.
    // It is safe to access m_idMap.size(), m_nameMap.size(), and m_propertyNames.size() because they don't chase pointers.
    return (m_idMap.size() + m_nameMap.size()) * sizeof(Element*) + m_propertyNames.size() * sizeof(AtomString);
}

inline ContainerNode& HTMLCollection::ownerNode() const
{
    return m_ownerNode;
}

inline Ref<ContainerNode> HTMLCollection::protectedOwnerNode() const
{
    return m_ownerNode;
}

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_HTMLCOLLECTION(ClassName, Type) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ClassName) \
    static bool isType(const WebCore::HTMLCollection& collection) { return collection.type() == WebCore::Type; } \
SPECIALIZE_TYPE_TRAITS_END()
