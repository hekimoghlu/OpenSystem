/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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
#include "HTMLBaseElement.h"

#include "Document.h"
#include "ElementInlines.h"
#include "HTMLNames.h"
#include "TextResourceDecoder.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLBaseElement);

using namespace HTMLNames;

inline HTMLBaseElement::HTMLBaseElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(baseTag));
}

Ref<HTMLBaseElement> HTMLBaseElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLBaseElement(tagName, document));
}

void HTMLBaseElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == hrefAttr || name == targetAttr) {
        if (isConnected())
            document().processBaseElement();
    } else
        HTMLElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

Node::InsertedIntoAncestorResult HTMLBaseElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    HTMLElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);
    if (insertionType.connectedToDocument)
        document().processBaseElement();
    return InsertedIntoAncestorResult::Done;
}

void HTMLBaseElement::removedFromAncestor(RemovalType removalType, ContainerNode& oldParentOfRemovedTree)
{
    HTMLElement::removedFromAncestor(removalType, oldParentOfRemovedTree);
    if (removalType.disconnectedFromDocument)
        document().processBaseElement();
}

bool HTMLBaseElement::isURLAttribute(const Attribute& attribute) const
{
    return attribute.name().localName() == hrefAttr || HTMLElement::isURLAttribute(attribute);
}

AtomString HTMLBaseElement::target() const
{
    return attributeWithoutSynchronization(targetAttr);
}

// https://html.spec.whatwg.org/#dom-base-href
String HTMLBaseElement::href() const
{
    AtomString url = attributeWithoutSynchronization(hrefAttr);
    if (url.isNull())
        url = emptyAtom();

    auto urlRecord = document().completeURL(url, document().fallbackBaseURL());
    if (!urlRecord.isValid())
        return url;

    return urlRecord.string();
}

void HTMLBaseElement::setHref(const AtomString& value)
{
    setAttributeWithoutSynchronization(hrefAttr, value);
}

}
