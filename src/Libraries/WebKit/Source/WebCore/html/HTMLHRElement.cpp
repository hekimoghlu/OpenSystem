/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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
#include "HTMLHRElement.h"

#include "CSSPropertyNames.h"
#include "CSSValueKeywords.h"
#include "CSSValuePool.h"
#include "ElementInlines.h"
#include "HTMLNames.h"
#include "HTMLParserIdioms.h"
#include "MutableStyleProperties.h"
#include "NodeName.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLHRElement);

using namespace HTMLNames;

HTMLHRElement::HTMLHRElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(hrTag));
}

Ref<HTMLHRElement> HTMLHRElement::create(Document& document)
{
    return adoptRef(*new HTMLHRElement(hrTag, document));
}

Ref<HTMLHRElement> HTMLHRElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLHRElement(tagName, document));
}

bool HTMLHRElement::hasPresentationalHintsForAttribute(const QualifiedName& name) const
{
    switch (name.nodeName()) {
    case AttributeNames::widthAttr:
    case AttributeNames::colorAttr:
    case AttributeNames::noshadeAttr:
    case AttributeNames::sizeAttr:
        return true;
    default:
        break;
    }
    return HTMLElement::hasPresentationalHintsForAttribute(name);
}

void HTMLHRElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
{
    switch (name.nodeName()) {
    case AttributeNames::alignAttr:
        if (equalLettersIgnoringASCIICase(value, "left"_s)) {
            addPropertyToPresentationalHintStyle(style, CSSPropertyMarginLeft, 0, CSSUnitType::CSS_PX);
            addPropertyToPresentationalHintStyle(style, CSSPropertyMarginRight, CSSValueAuto);
        } else if (equalLettersIgnoringASCIICase(value, "right"_s)) {
            addPropertyToPresentationalHintStyle(style, CSSPropertyMarginLeft, CSSValueAuto);
            addPropertyToPresentationalHintStyle(style, CSSPropertyMarginRight, 0, CSSUnitType::CSS_PX);
        } else {
            addPropertyToPresentationalHintStyle(style, CSSPropertyMarginLeft, CSSValueAuto);
            addPropertyToPresentationalHintStyle(style, CSSPropertyMarginRight, CSSValueAuto);
        }
        break;
    case AttributeNames::widthAttr:
        addHTMLLengthToStyle(style, CSSPropertyWidth, value);
        break;
    case AttributeNames::colorAttr:
        addPropertyToPresentationalHintStyle(style, CSSPropertyBorderStyle, CSSValueSolid);
        addHTMLColorToStyle(style, CSSPropertyBorderColor, value);
        addHTMLColorToStyle(style, CSSPropertyBackgroundColor, value);
        break;
    case AttributeNames::noshadeAttr:
        if (!hasAttributeWithoutSynchronization(colorAttr)) {
            addPropertyToPresentationalHintStyle(style, CSSPropertyBorderStyle, CSSValueSolid);
            auto darkGrayValue = CSSValuePool::singleton().createColorValue(Color::darkGray);
            style.setProperty(CSSPropertyBorderColor, darkGrayValue.ptr());
            style.setProperty(CSSPropertyBackgroundColor, WTFMove(darkGrayValue));
        }
        break;
    case AttributeNames::sizeAttr:
        if (int size = parseHTMLInteger(value).value_or(0); size > 1)
            addPropertyToPresentationalHintStyle(style, CSSPropertyHeight, size - 2, CSSUnitType::CSS_PX);
        else
            addPropertyToPresentationalHintStyle(style, CSSPropertyBorderBottomWidth, 0, CSSUnitType::CSS_PX);
        break;
    default:
        HTMLElement::collectPresentationalHintsForAttribute(name, value, style);
        break;
    }
}

bool HTMLHRElement::canContainRangeEndPoint() const
{
    return hasChildNodes() && HTMLElement::canContainRangeEndPoint();
}

}
