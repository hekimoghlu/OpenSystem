/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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
#include "HTMLNameCollection.h"

#include "Element.h"
#include "HTMLEmbedElement.h"
#include "HTMLFormElement.h"
#include "HTMLIFrameElement.h"
#include "HTMLImageElement.h"
#include "HTMLNameCollectionInlines.h"
#include "HTMLNames.h"
#include "HTMLObjectElement.h"
#include "NodeRareData.h"
#include "NodeTraversal.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace HTMLNames;

template HTMLNameCollection<WindowNameCollection, CollectionTraversalType::Descendants>::~HTMLNameCollection();
template HTMLNameCollection<DocumentNameCollection, CollectionTraversalType::Descendants>::~HTMLNameCollection();

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WindowNameCollection);
WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DocumentNameCollection);

bool WindowNameCollection::elementMatchesIfNameAttributeMatch(const Element& element)
{
    return is<HTMLEmbedElement>(element)
        || is<HTMLFormElement>(element)
        || is<HTMLImageElement>(element)
        || is<HTMLObjectElement>(element);
}

bool WindowNameCollection::elementMatches(const Element& element, const AtomString& name)
{
    // Find only images, forms, applets, embeds and objects by name, but anything by id.
    return (elementMatchesIfNameAttributeMatch(element) && element.getNameAttribute() == name)
        || element.getIdAttribute() == name;
}

static inline bool isObjectElementForDocumentNameCollection(const Element& element)
{
    auto* objectElement = dynamicDowncast<HTMLObjectElement>(element);
    return objectElement && objectElement->isExposed();
}

bool DocumentNameCollection::elementMatchesIfIdAttributeMatch(const Element& element)
{
    // FIXME: We need to fix HTMLImageElement to update the hash map for us when the name attribute is removed.
    return isObjectElementForDocumentNameCollection(element)
        || (is<HTMLImageElement>(element) && element.hasName() && !element.getNameAttribute().isEmpty());
}

bool DocumentNameCollection::elementMatchesIfNameAttributeMatch(const Element& element)
{
    return isObjectElementForDocumentNameCollection(element)
        || is<HTMLEmbedElement>(element)
        || is<HTMLFormElement>(element)
        || is<HTMLIFrameElement>(element)
        || is<HTMLImageElement>(element);
}

bool DocumentNameCollection::elementMatches(const Element& element, const AtomString& name)
{
    // Find images, forms, applets, embeds, objects and iframes by name, applets and object by id, and images by id
    // but only if they have a name attribute (this very strange rule matches IE).
    return (elementMatchesIfNameAttributeMatch(element) && element.getNameAttribute() == name)
        || (elementMatchesIfIdAttributeMatch(element) && element.getIdAttribute() == name);
}

}
