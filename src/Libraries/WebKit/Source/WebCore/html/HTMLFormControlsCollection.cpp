/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 2, 2023.
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
#include "HTMLFormControlsCollection.h"

#include "CachedHTMLCollectionInlines.h"
#include "FormListedElement.h"
#include "HTMLFormElement.h"
#include "HTMLImageElement.h"
#include "HTMLNames.h"
#include "ScriptDisallowedScope.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace HTMLNames;

// Since the collections are to be "live", we have to do the
// calculation every time if anything has changed.

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLFormControlsCollection);

HTMLFormControlsCollection::HTMLFormControlsCollection(ContainerNode& ownerNode)
    : CachedHTMLCollection(ownerNode, CollectionType::FormControls)
{
    ASSERT(is<HTMLFormElement>(ownerNode));
}

Ref<HTMLFormControlsCollection> HTMLFormControlsCollection::create(ContainerNode& ownerNode, CollectionType)
{
    return adoptRef(*new HTMLFormControlsCollection(ownerNode));
}

HTMLFormControlsCollection::~HTMLFormControlsCollection() = default;

std::optional<std::variant<RefPtr<RadioNodeList>, RefPtr<Element>>> HTMLFormControlsCollection::namedItemOrItems(const AtomString& name) const
{
    auto namedItems = this->namedItems(name);

    if (namedItems.isEmpty())
        return std::nullopt;
    if (namedItems.size() == 1)
        return std::variant<RefPtr<RadioNodeList>, RefPtr<Element>> { RefPtr<Element> { WTFMove(namedItems[0]) } };

    return std::variant<RefPtr<RadioNodeList>, RefPtr<Element>> { RefPtr<RadioNodeList> { ownerNode().radioNodeList(name) } };
}

static unsigned findFormListedElement(const Vector<WeakPtr<HTMLElement, WeakPtrImplWithEventTargetData>>& elements, const Element& element)
{
    for (unsigned i = 0; i < elements.size(); ++i) {
        RefPtr currentElement = elements[i].get();
        ASSERT(currentElement);
        auto* listedElement = currentElement->asFormListedElement();
        ASSERT(listedElement);
        if (listedElement->isEnumeratable() && currentElement == &element)
            return i;
    }
    return elements.size();
}

HTMLElement* HTMLFormControlsCollection::customElementAfter(Element* current) const
{
    ScriptDisallowedScope::InMainThread scriptDisallowedScope;
    auto& elements = ownerNode().unsafeListedElements();
    unsigned start;
    if (!current)
        start = 0;
    else if (m_cachedElement == current)
        start = m_cachedElementOffsetInArray + 1;
    else
        start = findFormListedElement(elements, *current) + 1;

    for (unsigned i = start; i < elements.size(); ++i) {
        RefPtr element = elements[i].get();
        ASSERT(element);
        ASSERT(element->asFormListedElement());
        if (element->asFormListedElement()->isEnumeratable()) {
            m_cachedElement = element.get();
            m_cachedElementOffsetInArray = i;
            return element.get();
        }
    }
    return nullptr;
}

HTMLFormElement& HTMLFormControlsCollection::ownerNode() const
{
    return downcast<HTMLFormElement>(CachedHTMLCollection::ownerNode());
}

void HTMLFormControlsCollection::updateNamedElementCache() const
{
    if (hasNamedElementCache())
        return;

    auto cache = makeUnique<CollectionNamedElementCache>();

    UncheckedKeyHashSet<AtomString> foundInputElements;

    ScriptDisallowedScope::InMainThread scriptDisallowedScope;
    for (auto& weakElement : ownerNode().unsafeListedElements()) {
        RefPtr element { weakElement.get() };
        ASSERT(element);
        auto* associatedElement = element->asFormListedElement();
        ASSERT(associatedElement);
        if (associatedElement->isEnumeratable()) {
            const AtomString& id = element->getIdAttribute();
            if (!id.isEmpty()) {
                cache->appendToIdCache(id, *element);
                foundInputElements.add(id);
            }
            const AtomString& name = element->getNameAttribute();
            if (!name.isEmpty() && id != name) {
                cache->appendToNameCache(name, *element);
                foundInputElements.add(name);
            }
        }
    }

    for (auto& elementPtr : ownerNode().imageElements()) {
        if (!elementPtr)
            continue;
        HTMLImageElement& element = *elementPtr;
        const AtomString& id = element.getIdAttribute();
        if (!id.isEmpty() && !foundInputElements.contains(id))
            cache->appendToIdCache(id, element);
        const AtomString& name = element.getNameAttribute();
        if (!name.isEmpty() && id != name && !foundInputElements.contains(name))
            cache->appendToNameCache(name, element);
    }

    setNamedItemCache(WTFMove(cache));
}

void HTMLFormControlsCollection::invalidateCacheForDocument(Document& document)
{
    CachedHTMLCollection::invalidateCacheForDocument(document);
    m_cachedElement = nullptr;
    m_cachedElementOffsetInArray = 0;
}

HTMLElement* HTMLFormControlsCollection::item(unsigned offset) const
{
    return downcast<HTMLElement>(CachedHTMLCollection::item(offset));
}

}
