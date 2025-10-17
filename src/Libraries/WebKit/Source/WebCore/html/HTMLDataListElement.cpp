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
#include "config.h"

#include "HTMLDataListElement.h"

#include "GenericCachedHTMLCollection.h"
#include "HTMLNames.h"
#include "HTMLOptionElement.h"
#include "IdTargetObserverRegistry.h"
#include "NodeRareData.h"
#include "TypedElementDescendantIteratorInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLDataListElement);

inline HTMLDataListElement::HTMLDataListElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document, TypeFlag::HasDidMoveToNewDocument)
{
    document.incrementDataListElementCount();
}

HTMLDataListElement::~HTMLDataListElement()
{
    document().decrementDataListElementCount();
}

Ref<HTMLDataListElement> HTMLDataListElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLDataListElement(tagName, document));
}

void HTMLDataListElement::didMoveToNewDocument(Document& oldDocument, Document& newDocument)
{
    oldDocument.decrementDataListElementCount();
    newDocument.incrementDataListElementCount();
    HTMLElement::didMoveToNewDocument(oldDocument, newDocument);
}

Ref<HTMLCollection> HTMLDataListElement::options()
{
    return ensureRareData().ensureNodeLists().addCachedCollection<GenericCachedHTMLCollection<CollectionTypeTraits<CollectionType::DataListOptions>::traversalType>>(*this, CollectionType::DataListOptions);
}

void HTMLDataListElement::childrenChanged(const ChildChange& change)
{
    HTMLElement::childrenChanged(change);
    if (change.source == ChildChange::Source::API)
        optionElementChildrenChanged();
}

void HTMLDataListElement::optionElementChildrenChanged()
{
    if (auto& id = getIdAttribute(); !id.isEmpty()) {
        if (CheckedPtr observerRegistry = treeScope().idTargetObserverRegistryIfExists())
            observerRegistry->notifyObservers(*this, id);
    }
}

auto HTMLDataListElement::suggestions() const -> SuggestionRange
{
    return filteredDescendants<HTMLOptionElement, isSuggestion>(*this);
}

bool HTMLDataListElement::isSuggestion(const HTMLOptionElement& descendant)
{
    return !descendant.isDisabledFormControl() && !descendant.value().isEmpty();
}

} // namespace WebCore
