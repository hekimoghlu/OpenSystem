/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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
#include "TextPlaceholderElement.h"

#include "HTMLNames.h"
#include "HTMLTextFormControlElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(TextPlaceholderElement);

Ref<TextPlaceholderElement> TextPlaceholderElement::create(Document& document, const LayoutSize& size)
{
    return adoptRef(*new TextPlaceholderElement { document, size });
}

TextPlaceholderElement::TextPlaceholderElement(Document& document, const LayoutSize& size)
    : HTMLDivElement { HTMLNames::divTag, document }
{
    // FIXME: Move to User Agent stylesheet. See <https://webkit.org/b/208745>.
    setInlineStyleProperty(CSSPropertyDisplay, size.width() ? CSSValueInlineBlock : CSSValueBlock);
    setInlineStyleProperty(CSSPropertyVerticalAlign, CSSValueTop);
    setInlineStyleProperty(CSSPropertyVisibility, CSSValueHidden, IsImportant::Yes);
    if (size.width())
        setInlineStyleProperty(CSSPropertyWidth, size.width(), CSSUnitType::CSS_PX);
    setInlineStyleProperty(CSSPropertyHeight, size.height(), CSSUnitType::CSS_PX);
}

auto TextPlaceholderElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree) -> InsertedIntoAncestorResult
{
    if (insertionType.treeScopeChanged) {
        if (RefPtr shadowHost = dynamicDowncast<HTMLTextFormControlElement>(parentOfInsertedTree.shadowHost()))
            shadowHost->setCanShowPlaceholder(false);
    }
    return HTMLDivElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);
}

void TextPlaceholderElement::removedFromAncestor(RemovalType removalType, ContainerNode& oldParentOfRemovedTree)
{
    if (removalType.treeScopeChanged) {
        if (RefPtr shadowHost = dynamicDowncast<HTMLTextFormControlElement>(oldParentOfRemovedTree.shadowHost()))
            shadowHost->setCanShowPlaceholder(true);
    }
    HTMLDivElement::removedFromAncestor(removalType, oldParentOfRemovedTree);
}

} // namespace WebCore
