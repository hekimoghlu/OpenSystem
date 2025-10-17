/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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
#include "HTMLArticleElement.h"

#include "HTMLDocument.h"
#include "HTMLNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLArticleElement);

Ref<HTMLArticleElement> HTMLArticleElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLArticleElement(tagName, document));
}

HTMLArticleElement::HTMLArticleElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(tagName == HTMLNames::articleTag);
}

auto HTMLArticleElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree) -> InsertedIntoAncestorResult
{
    auto result = HTMLElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);

    if (insertionType.connectedToDocument) {
        if (auto* newDocument = dynamicDowncast<HTMLDocument>(parentOfInsertedTree.document()))
            newDocument->registerArticleElement(*this);
    }

    return result;
}

void HTMLArticleElement::removedFromAncestor(RemovalType removalType, ContainerNode& oldParentOfRemovedTree)
{
    if (removalType.disconnectedFromDocument)
        oldParentOfRemovedTree.document().unregisterArticleElement(*this);

    HTMLElement::removedFromAncestor(removalType, oldParentOfRemovedTree);
}

} // namespace WebCore
