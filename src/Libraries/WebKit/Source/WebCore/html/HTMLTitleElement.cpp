/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 4, 2023.
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
#include "HTMLTitleElement.h"

#include "Document.h"
#include "ElementInlines.h"
#include "HTMLNames.h"
#include "NodeRenderStyle.h"
#include "RenderElement.h"
#include "RenderStyle.h"
#include "ResolvedStyle.h"
#include "StyleInheritedData.h"
#include "StyleResolver.h"
#include "Text.h"
#include "TextManipulationController.h"
#include "TextNodeTraversal.h"
#include <wtf/Ref.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLTitleElement);

using namespace HTMLNames;

inline HTMLTitleElement::HTMLTitleElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(titleTag));
}

Ref<HTMLTitleElement> HTMLTitleElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLTitleElement(tagName, document));
}

Node::InsertedIntoAncestorResult HTMLTitleElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    HTMLElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);

    if (insertionType.connectedToDocument) {
        m_title = computedTextWithDirection();
        document().titleElementAdded(*this);
    }
    return InsertedIntoAncestorResult::Done;
}

void HTMLTitleElement::removedFromAncestor(RemovalType removalType, ContainerNode& oldParentOfRemovedTree)
{
    HTMLElement::removedFromAncestor(removalType, oldParentOfRemovedTree);

    if (removalType.disconnectedFromDocument)
        document().titleElementRemoved(*this);
}

void HTMLTitleElement::childrenChanged(const ChildChange& change)
{
    HTMLElement::childrenChanged(change);

    if (isConnected()) {
        m_title = computedTextWithDirection();
        document().titleElementTextChanged(*this);
    }
}

String HTMLTitleElement::text() const
{
    return TextNodeTraversal::childTextContent(*this);
}

StringWithDirection HTMLTitleElement::computedTextWithDirection()
{
    ASSERT(isConnected());
    if (!firstChild())
        return { };
    auto direction = TextDirection::LTR;
    if (auto* computedStyle = this->computedStyle())
        direction = computedStyle->writingMode().computedTextDirection();
    return { text(), direction };
}

void HTMLTitleElement::setText(String&& value)
{
    setTextContent(WTFMove(value));
}

}
