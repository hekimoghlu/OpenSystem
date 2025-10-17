/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 20, 2023.
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
#include "config.h"
#include "HTMLAllCollection.h"

#include "CachedHTMLCollectionInlines.h"
#include "Element.h"
#include "NodeRareDataInlines.h"
#include <JavaScriptCore/Identifier.h>
#include <variant>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLAllNamedSubCollection);

Ref<HTMLAllCollection> HTMLAllCollection::create(Document& document, CollectionType type)
{
    return adoptRef(*new HTMLAllCollection(document, type));
}

inline HTMLAllCollection::HTMLAllCollection(Document& document, CollectionType type)
    : AllDescendantsCollection(document, type)
{
}

// https://html.spec.whatwg.org/multipage/infrastructure.html#dom-htmlallcollection-item
std::optional<std::variant<RefPtr<HTMLCollection>, RefPtr<Element>>> HTMLAllCollection::namedOrIndexedItemOrItems(const AtomString& nameOrIndex) const
{
    if (nameOrIndex.isNull())
        return std::nullopt;

    if (auto index = JSC::parseIndex(*nameOrIndex.impl()))
        return std::variant<RefPtr<HTMLCollection>, RefPtr<Element>> { RefPtr<Element> { item(index.value()) } };

    return namedItemOrItems(nameOrIndex);
}

// https://html.spec.whatwg.org/multipage/infrastructure.html#concept-get-all-named
std::optional<std::variant<RefPtr<HTMLCollection>, RefPtr<Element>>> HTMLAllCollection::namedItemOrItems(const AtomString& name) const
{
    auto namedItems = this->namedItems(name);

    if (namedItems.isEmpty())
        return std::nullopt;
    if (namedItems.size() == 1)
        return std::variant<RefPtr<HTMLCollection>, RefPtr<Element>> { RefPtr<Element> { WTFMove(namedItems[0]) } };

    return std::variant<RefPtr<HTMLCollection>, RefPtr<Element>> { RefPtr<HTMLCollection> { downcast<Document>(ownerNode()).allFilteredByName(name) } };
}

HTMLAllNamedSubCollection::HTMLAllNamedSubCollection(Document& document, CollectionType type, const AtomString& name)
    : CachedHTMLCollection(document, type)
    , m_name(name)
{
    ASSERT(type == CollectionType::DocumentAllNamedItems);
}

HTMLAllNamedSubCollection::~HTMLAllNamedSubCollection()
{
    ownerNode().nodeLists()->removeCachedCollection(this, m_name);
}

bool HTMLAllNamedSubCollection::elementMatches(Element& element) const
{
    const auto& id = element.getIdAttribute();
    if (id == m_name)
        return true;

    if (!nameShouldBeVisibleInDocumentAll(element))
        return false;

    const auto& name = element.getNameAttribute();
    if (name == m_name)
        return true;

    return false;
}

} // namespace WebCore
