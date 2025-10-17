/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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
#include "NodeRareData.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class Document;

template <typename HTMLCollectionClass, CollectionTraversalType traversalType>
class HTMLNameCollection : public CachedHTMLCollection<HTMLCollectionClass, traversalType> {
    WTF_MAKE_TZONE_OR_ISO_NON_HEAP_ALLOCATABLE(HTMLNameCollection);
public:
    virtual ~HTMLNameCollection();

    Document& document() { return downcast<Document>(this->ownerNode()); }

protected:
    HTMLNameCollection(Document&, CollectionType, const AtomString& name);

    AtomString m_name;
};

template <typename HTMLCollectionClass, CollectionTraversalType traversalType>
HTMLNameCollection<HTMLCollectionClass, traversalType>::HTMLNameCollection(Document& document, CollectionType type, const AtomString& name)
    : CachedHTMLCollection<HTMLCollectionClass, traversalType>(document, type)
    , m_name(name)
{
}

class WindowNameCollection final : public HTMLNameCollection<WindowNameCollection, CollectionTraversalType::Descendants> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WindowNameCollection);
public:
    static Ref<WindowNameCollection> create(Document& document, CollectionType type, const AtomString& name)
    {
        return adoptRef(*new WindowNameCollection(document, type, name));
    }

    // For CachedHTMLCollection.
    bool elementMatches(const Element& element) const { return elementMatches(element, m_name.impl()); }

    static bool elementMatchesIfIdAttributeMatch(const Element&) { return true; }
    static bool elementMatchesIfNameAttributeMatch(const Element&);
    static bool elementMatches(const Element&, const AtomString&);

private:
    WindowNameCollection(Document& document, CollectionType type, const AtomString& name)
        : HTMLNameCollection<WindowNameCollection, CollectionTraversalType::Descendants>(document, type, name)
    {
        ASSERT(type == CollectionType::WindowNamedItems);
    }
};

class DocumentNameCollection final : public HTMLNameCollection<DocumentNameCollection, CollectionTraversalType::Descendants> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DocumentNameCollection);
public:
    static Ref<DocumentNameCollection> create(Document& document, CollectionType type, const AtomString& name)
    {
        return adoptRef(*new DocumentNameCollection(document, type, name));
    }

    static bool elementMatchesIfIdAttributeMatch(const Element&);
    static bool elementMatchesIfNameAttributeMatch(const Element&);

    // For CachedHTMLCollection.
    bool elementMatches(const Element& element) const { return elementMatches(element, m_name.impl()); }

    static bool elementMatches(const Element&, const AtomString&);

private:
    DocumentNameCollection(Document& document, CollectionType type, const AtomString& name)
        : HTMLNameCollection<DocumentNameCollection, CollectionTraversalType::Descendants>(document, type, name)
    {
        ASSERT(type == CollectionType::DocumentNamedItems);
    }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_HTMLCOLLECTION(WindowNameCollection, CollectionType::WindowNamedItems)
SPECIALIZE_TYPE_TRAITS_HTMLCOLLECTION(DocumentNameCollection, CollectionType::DocumentNamedItems)
