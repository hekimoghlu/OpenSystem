/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 5, 2025.
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
#include "HTMLFontElement.h"

#include "CSSPropertyNames.h"
#include "CSSStyleSheet.h"
#include "CSSValueKeywords.h"
#include "CSSValueList.h"
#include "CSSValuePool.h"
#include "HTMLNames.h"
#include "MutableStyleProperties.h"
#include "NodeName.h"
#include "StyleProperties.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/ParsingUtilities.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLFontElement);

using namespace HTMLNames;

HTMLFontElement::HTMLFontElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(fontTag));
}

Ref<HTMLFontElement> HTMLFontElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLFontElement(tagName, document));
}

// http://www.whatwg.org/specs/web-apps/current-work/multipage/rendering.html#fonts-and-colors
template <typename CharacterType>
static bool parseFontSize(std::span<const CharacterType> characters, int& size)
{

    // Step 1
    // Step 2
    // Step 3
    while (!characters.empty()) {
        if (!skipExactly<isASCIIWhitespace>(characters))
            break;
    }

    // Step 4
    if (characters.empty())
        return false;

    // Step 5
    enum {
        RelativePlus,
        RelativeMinus,
        Absolute
    } mode;

    switch (characters.front()) {
    case '+':
        mode = RelativePlus;
        skip(characters, 1);
        break;
    case '-':
        mode = RelativeMinus;
        skip(characters, 1);
        break;
    default:
        mode = Absolute;
        break;
    }

    // Step 6
    StringBuilder digits;
    digits.reserveCapacity(16);
    while (!characters.empty()) {
        if (!isASCIIDigit(characters.front()))
            break;
        digits.append(consume(characters));
    }

    // Step 7
    if (digits.isEmpty())
        return false;

    // Step 8
    int value = parseInteger<int>(digits).value_or(0);

    // Step 9
    if (mode == RelativePlus)
        value += 3;
    else if (mode == RelativeMinus)
        value = 3 - value;

    // Step 10
    if (value > 7)
        value = 7;

    // Step 11
    if (value < 1)
        value = 1;

    size = value;
    return true;
}

static bool parseFontSize(const String& input, int& size)
{
    if (input.isEmpty())
        return false;

    if (input.is8Bit())
        return parseFontSize(input.span8(), size);

    return parseFontSize(input.span16(), size);
}

bool HTMLFontElement::cssValueFromFontSizeNumber(const String& s, CSSValueID& size)
{
    int num = 0;
    if (!parseFontSize(s, num))
        return false;

    switch (num) {
    case 1:
        // FIXME: The spec says that we're supposed to use CSSValueXxSmall here.
        size = CSSValueXSmall;
        break;
    case 2: 
        size = CSSValueSmall;
        break;
    case 3: 
        size = CSSValueMedium;
        break;
    case 4: 
        size = CSSValueLarge;
        break;
    case 5: 
        size = CSSValueXLarge;
        break;
    case 6: 
        size = CSSValueXxLarge;
        break;
    case 7:
        size = CSSValueXxxLarge;
        break;
    default:
        ASSERT_NOT_REACHED();
    }
    return true;
}

bool HTMLFontElement::hasPresentationalHintsForAttribute(const QualifiedName& name) const
{
    switch (name.nodeName()) {
    case AttributeNames::sizeAttr:
    case AttributeNames::colorAttr:
    case AttributeNames::faceAttr:
        return true;
    default:
        break;
    }
    return HTMLElement::hasPresentationalHintsForAttribute(name);
}

void HTMLFontElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
{
    switch (name.nodeName()) {
    case AttributeNames::sizeAttr: {
        CSSValueID size = CSSValueInvalid;
        if (cssValueFromFontSizeNumber(value, size))
            addPropertyToPresentationalHintStyle(style, CSSPropertyFontSize, size);
        break;
    }
    case AttributeNames::colorAttr:
        addHTMLColorToStyle(style, CSSPropertyColor, value);
        break;
    case AttributeNames::faceAttr:
        if (!value.isEmpty()) {
            if (auto fontFaceValue = CSSValuePool::singleton().createFontFaceValue(value))
                style.setProperty(CSSProperty(CSSPropertyFontFamily, WTFMove(fontFaceValue)));
        }
        break;
    default:
        HTMLElement::collectPresentationalHintsForAttribute(name, value, style);
        break;
    }
}

}
