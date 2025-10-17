/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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
#include "HTMLMapElement.h"

#include "Attribute.h"
#include "Document.h"
#include "ElementInlines.h"
#include "GenericCachedHTMLCollection.h"
#include "HTMLAreaElement.h"
#include "HTMLImageElement.h"
#include "HitTestResult.h"
#include "IntSize.h"
#include "NodeRareData.h"
#include "TypedElementDescendantIteratorInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLMapElement);

using namespace HTMLNames;

HTMLMapElement::HTMLMapElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(mapTag));
}

Ref<HTMLMapElement> HTMLMapElement::create(Document& document)
{
    return adoptRef(*new HTMLMapElement(mapTag, document));
}

Ref<HTMLMapElement> HTMLMapElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLMapElement(tagName, document));
}

HTMLMapElement::~HTMLMapElement() = default;

bool HTMLMapElement::mapMouseEvent(LayoutPoint location, const LayoutSize& size, HitTestResult& result)
{
    RefPtr<HTMLAreaElement> defaultArea;

    for (auto& area : descendantsOfType<HTMLAreaElement>(*this)) {
        if (area.isDefault()) {
            if (!defaultArea)
                defaultArea = &area;
        } else if (area.mapMouseEvent(location, size, result))
            return true;
    }
    
    if (defaultArea) {
        result.setInnerNode(defaultArea.get());
        result.setURLElement(defaultArea.get());
    }
    return defaultArea;
}

RefPtr<HTMLImageElement> HTMLMapElement::imageElement()
{
    if (m_name.isEmpty())
        return nullptr;
    return treeScope().imageElementByUsemap(m_name);
}

void HTMLMapElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    HTMLElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);

    // FIXME: This logic seems wrong for XML documents.
    // Either the id or name will be used depending on the order the attributes are parsed.

    if (name == HTMLNames::idAttr || name == HTMLNames::nameAttr) {
        if (name == HTMLNames::idAttr) {
            // Call base class so that hasID bit gets set.
            HTMLElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
            if (document().isHTMLDocument())
                return;
        }
        if (isInTreeScope())
            treeScope().removeImageMap(*this);
        AtomString mapName = newValue;
        if (mapName[0] == '#')
            mapName = StringView(mapName).substring(1).toAtomString();
        m_name = WTFMove(mapName);
        if (isInTreeScope())
            treeScope().addImageMap(*this);
    }
}

Ref<HTMLCollection> HTMLMapElement::areas()
{
    return ensureRareData().ensureNodeLists().addCachedCollection<GenericCachedHTMLCollection<CollectionTypeTraits<CollectionType::MapAreas>::traversalType>>(*this, CollectionType::MapAreas);
}

Node::InsertedIntoAncestorResult HTMLMapElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    Node::InsertedIntoAncestorResult request = HTMLElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);
    if (insertionType.treeScopeChanged)
        treeScope().addImageMap(*this);
    return request;
}

void HTMLMapElement::removedFromAncestor(RemovalType removalType, ContainerNode& oldParentOfRemovedTree)
{
    if (removalType.treeScopeChanged)
        oldParentOfRemovedTree.treeScope().removeImageMap(*this);
    HTMLElement::removedFromAncestor(removalType, oldParentOfRemovedTree);
}

}
