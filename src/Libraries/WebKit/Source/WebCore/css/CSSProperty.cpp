/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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
#include "CSSProperty.h"

#include "CSSValueList.h"
#include "StylePropertyShorthand.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

struct SameSizeAsCSSProperty {
    uint32_t bitfields;
    void* value;
};

static_assert(sizeof(CSSProperty) == sizeof(SameSizeAsCSSProperty), "CSSProperty should stay small");

CSSPropertyID StylePropertyMetadata::shorthandID() const
{
    if (!m_isSetFromShorthand)
        return CSSPropertyInvalid;

    auto shorthands = matchingShorthandsForLonghand(static_cast<CSSPropertyID>(m_propertyID));
    ASSERT(shorthands.size() && m_indexInShorthandsVector >= 0 && m_indexInShorthandsVector < shorthands.size());
    return shorthands[m_indexInShorthandsVector].id();
}

bool CSSProperty::isInsetProperty(CSSPropertyID propertyID)
{
    switch (propertyID) {
    case CSSPropertyInset:
    case CSSPropertyLeft:
    case CSSPropertyRight:
    case CSSPropertyTop:
    case CSSPropertyBottom:

    case CSSPropertyInsetInline:
    case CSSPropertyInsetInlineStart:
    case CSSPropertyInsetInlineEnd:

    case CSSPropertyInsetBlock:
    case CSSPropertyInsetBlockStart:
    case CSSPropertyInsetBlockEnd:
        return true;
    default:
        return false;
    }
};

bool CSSProperty::isSizingProperty(CSSPropertyID propertyID)
{
    switch (propertyID) {
    case CSSPropertyWidth:
    case CSSPropertyMinWidth:
    case CSSPropertyMaxWidth:

    case CSSPropertyHeight:
    case CSSPropertyMinHeight:
    case CSSPropertyMaxHeight:

    case CSSPropertyBlockSize:
    case CSSPropertyMinBlockSize:
    case CSSPropertyMaxBlockSize:

    case CSSPropertyInlineSize:
    case CSSPropertyMinInlineSize:
    case CSSPropertyMaxInlineSize:
        return true;
    default:
        return false;
    }
}

bool CSSProperty::isMarginProperty(CSSPropertyID propertyID)
{
    switch (propertyID) {
    case CSSPropertyMargin:
    case CSSPropertyMarginLeft:
    case CSSPropertyMarginRight:
    case CSSPropertyMarginTop:
    case CSSPropertyMarginBottom:

    case CSSPropertyMarginBlock:
    case CSSPropertyMarginBlockStart:
    case CSSPropertyMarginBlockEnd:

    case CSSPropertyMarginInline:
    case CSSPropertyMarginInlineStart:
    case CSSPropertyMarginInlineEnd:
        return true;

    default:
        return false;
    }
}


} // namespace WebCore
