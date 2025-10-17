/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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
#include "SVGTitleElement.h"

#include "Document.h"
#include "SVGElementTypeHelpers.h"
#include "SVGSVGElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGTitleElement);

inline SVGTitleElement::SVGTitleElement(const QualifiedName& tagName, Document& document)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::titleTag));
}

Ref<SVGTitleElement> SVGTitleElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGTitleElement(tagName, document));
}

Node::InsertedIntoAncestorResult SVGTitleElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    auto result = SVGElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);
    if (insertionType.connectedToDocument && parentNode() == document().documentElement())
        protectedDocument()->titleElementAdded(*this);
    return result;
}

static bool isTitleElementRemovedFromSVGSVGElement(SVGTitleElement& title, ContainerNode& oldParentOfRemovedTree)
{
    if (!title.parentNode() && is<SVGSVGElement>(oldParentOfRemovedTree) && title.document().documentElement() == &oldParentOfRemovedTree)
        return true;
    if (title.parentNode() && is<SVGSVGElement>(*title.parentNode()) && !title.parentNode()->parentNode() && is<Document>(oldParentOfRemovedTree))
        return true;
    return false;
}

void SVGTitleElement::removedFromAncestor(RemovalType removalType, ContainerNode& oldParentOfRemovedTree)
{
    SVGElement::removedFromAncestor(removalType, oldParentOfRemovedTree);
    if (removalType.disconnectedFromDocument && isTitleElementRemovedFromSVGSVGElement(*this, oldParentOfRemovedTree)) {
        Ref<Document> document = this->document();
        document->titleElementRemoved(*this);
    }
}

void SVGTitleElement::childrenChanged(const ChildChange& change)
{
    SVGElement::childrenChanged(change);
    if (isConnected() && parentNode() == document().documentElement())
        protectedDocument()->titleElementTextChanged(*this);
}

} // namespace WebCore
