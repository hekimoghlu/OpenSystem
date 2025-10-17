/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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

#include "AllDescendantsCollection.h"

namespace WebCore {

class HTMLAllCollection final : public AllDescendantsCollection {
public:
    static Ref<HTMLAllCollection> create(Document&, CollectionType);

    std::optional<std::variant<RefPtr<HTMLCollection>, RefPtr<Element>>> namedOrIndexedItemOrItems(const AtomString& nameOrIndex) const;
    std::optional<std::variant<RefPtr<HTMLCollection>, RefPtr<Element>>> namedItemOrItems(const AtomString&) const;

private:
    HTMLAllCollection(Document&, CollectionType);
};
static_assert(sizeof(HTMLAllCollection) == sizeof(AllDescendantsCollection));

class HTMLAllNamedSubCollection final : public CachedHTMLCollection<HTMLAllNamedSubCollection, CollectionTraversalType::Descendants> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLAllNamedSubCollection);
public:
    static Ref<HTMLAllNamedSubCollection> create(Document& document, CollectionType type, const AtomString& name)
    {
        return adoptRef(*new HTMLAllNamedSubCollection(document, type, name));
    }
    virtual ~HTMLAllNamedSubCollection();

    bool elementMatches(Element&) const;

private:
    HTMLAllNamedSubCollection(Document&, CollectionType, const AtomString&);

    AtomString m_name;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_HTMLCOLLECTION(HTMLAllCollection, CollectionType::DocAll)
SPECIALIZE_TYPE_TRAITS_HTMLCOLLECTION(HTMLAllNamedSubCollection, CollectionType::DocumentAllNamedItems)
