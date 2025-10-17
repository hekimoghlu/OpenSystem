/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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
#include "SVGStyleElement.h"

#include "CSSStyleSheet.h"
#include "CommonAtomStrings.h"
#include "Document.h"
#include "NodeName.h"
#include "SVGElementInlines.h"
#include "SVGNames.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGStyleElement);

inline SVGStyleElement::SVGStyleElement(const QualifiedName& tagName, Document& document, bool createdByParser)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
    , m_styleSheetOwner(document, createdByParser)
    , m_loadEventTimer(*this, &SVGElement::loadEventTimerFired)
{
    ASSERT(hasTagName(SVGNames::styleTag));
}

SVGStyleElement::~SVGStyleElement()
{
    m_styleSheetOwner.clearDocumentData(*this);
}

Ref<SVGStyleElement> SVGStyleElement::create(const QualifiedName& tagName, Document& document, bool createdByParser)
{
    return adoptRef(*new SVGStyleElement(tagName, document, createdByParser));
}

bool SVGStyleElement::disabled() const
{
    return sheet() && sheet()->disabled();
}

void SVGStyleElement::setDisabled(bool setDisabled)
{
    if (RefPtr styleSheet = sheet())
        styleSheet->setDisabled(setDisabled);
}

const AtomString& SVGStyleElement::type() const
{
    auto& typeValue = getAttribute(SVGNames::typeAttr);
    return typeValue.isNull() ? cssContentTypeAtom() : typeValue;
}

void SVGStyleElement::setType(const AtomString& type)
{
    setAttribute(SVGNames::typeAttr, type);
}

const AtomString& SVGStyleElement::media() const
{
    auto& value = attributeWithoutSynchronization(SVGNames::mediaAttr);
    return value.isNull() ? allAtom() : value;
}

void SVGStyleElement::setMedia(const AtomString& media)
{
    setAttributeWithoutSynchronization(SVGNames::mediaAttr, media);
}

String SVGStyleElement::title() const
{
    return attributeWithoutSynchronization(SVGNames::titleAttr);
}

void SVGStyleElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::titleAttr:
        if (RefPtr sheet = this->sheet(); sheet && !isInShadowTree())
            sheet->setTitle(newValue);
        break;
    case AttributeNames::typeAttr:
        m_styleSheetOwner.setContentType(newValue);
        break;
    case AttributeNames::mediaAttr:
        m_styleSheetOwner.setMedia(newValue);
        break;
    default:
        break;
    }

    SVGElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGStyleElement::finishParsingChildren()
{
    m_styleSheetOwner.finishParsingChildren(*this);
    SVGElement::finishParsingChildren();
}

Node::InsertedIntoAncestorResult SVGStyleElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    auto result = SVGElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);
    if (insertionType.connectedToDocument)
        m_styleSheetOwner.insertedIntoDocument(*this);
    return result;
}

void SVGStyleElement::removedFromAncestor(RemovalType removalType, ContainerNode& oldParentOfRemovedTree)
{
    SVGElement::removedFromAncestor(removalType, oldParentOfRemovedTree);
    if (removalType.disconnectedFromDocument)
        m_styleSheetOwner.removedFromDocument(*this);
}

void SVGStyleElement::childrenChanged(const ChildChange& change)
{
    SVGElement::childrenChanged(change);
    m_styleSheetOwner.childrenChanged(*this);
}

}
