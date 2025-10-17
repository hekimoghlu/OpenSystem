/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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

#include "CachedHTMLCollection.h"
#include "CommonAtomStrings.h"
#include <wtf/text/AtomString.h>

namespace WebCore {

// HTMLCollection that limits to a particular tag.
class TagCollection final : public CachedHTMLCollection<TagCollection, CollectionTypeTraits<CollectionType::ByTag>::traversalType> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TagCollection);
public:
    static Ref<TagCollection> create(ContainerNode& rootNode, CollectionType type, const AtomString& qualifiedName)
    {
        ASSERT_UNUSED(type, type == CollectionType::ByTag);
        return adoptRef(*new TagCollection(rootNode, qualifiedName));
    }

    virtual ~TagCollection();
    bool elementMatches(Element&) const;

private:
    TagCollection(ContainerNode& rootNode, const AtomString& qualifiedName);

    AtomString m_qualifiedName;
};

class TagCollectionNS final : public CachedHTMLCollection<TagCollectionNS, CollectionTypeTraits<CollectionType::ByTag>::traversalType> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TagCollectionNS);
public:
    static Ref<TagCollectionNS> create(ContainerNode& rootNode, const AtomString& namespaceURI, const AtomString& localName)
    {
        return adoptRef(*new TagCollectionNS(rootNode, namespaceURI, localName));
    }

    virtual ~TagCollectionNS();
    bool elementMatches(Element&) const;

private:
    TagCollectionNS(ContainerNode& rootNode, const AtomString& namespaceURI, const AtomString& localName);

    AtomString m_namespaceURI;
    AtomString m_localName;
};

class HTMLTagCollection final : public CachedHTMLCollection<HTMLTagCollection, CollectionTypeTraits<CollectionType::ByHTMLTag>::traversalType> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLTagCollection);
public:
    static Ref<HTMLTagCollection> create(ContainerNode& rootNode, CollectionType type, const AtomString& qualifiedName)
    {
        ASSERT_UNUSED(type, type == CollectionType::ByHTMLTag);
        return adoptRef(*new HTMLTagCollection(rootNode, qualifiedName));
    }

    virtual ~HTMLTagCollection();
    bool elementMatches(Element&) const;

private:
    HTMLTagCollection(ContainerNode& rootNode, const AtomString& qualifiedName);

    AtomString m_qualifiedName;
    AtomString m_loweredQualifiedName;
};

inline bool TagCollection::elementMatches(Element& element) const
{
    return m_qualifiedName == element.tagQName().toString();
}

inline bool TagCollectionNS::elementMatches(Element& element) const
{
    if (m_localName != starAtom() && m_localName != element.localName())
        return false;
    return m_namespaceURI == starAtom() || m_namespaceURI == element.namespaceURI();
}

inline bool HTMLTagCollection::elementMatches(Element& element) const
{
    if (element.isHTMLElement())
        return m_loweredQualifiedName == element.tagQName().toString();
    return m_qualifiedName == element.tagQName().toString();
}

} // namespace WebCore
