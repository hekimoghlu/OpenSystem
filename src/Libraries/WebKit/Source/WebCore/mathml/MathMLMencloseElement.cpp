/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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
#include "MathMLMencloseElement.h"

#if ENABLE(MATHML)

#include "ElementInlines.h"
#include "MathMLNames.h"
#include "RenderMathMLMenclose.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MathMLMencloseElement);

using namespace MathMLNames;

MathMLMencloseElement::MathMLMencloseElement(const QualifiedName& tagName, Document& document)
    : MathMLRowElement(tagName, document)
{
    // By default we draw a longdiv.
    clearNotations();
    addNotation(LongDiv);
}

Ref<MathMLMencloseElement> MathMLMencloseElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new MathMLMencloseElement(tagName, document));
}

RenderPtr<RenderElement> MathMLMencloseElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    return createRenderer<RenderMathMLMenclose>(*this, WTFMove(style));
}

void MathMLMencloseElement::addNotationFlags(StringView notation)
{
    ASSERT(m_notationFlags);
    if (notation == "longdiv"_s) {
        addNotation(LongDiv);
    } else if (notation == "roundedbox"_s) {
        addNotation(RoundedBox);
    } else if (notation == "circle"_s) {
        addNotation(Circle);
    } else if (notation == "left"_s) {
        addNotation(Left);
    } else if (notation == "right"_s) {
        addNotation(Right);
    } else if (notation == "top"_s) {
        addNotation(Top);
    } else if (notation == "bottom"_s) {
        addNotation(Bottom);
    } else if (notation == "updiagonalstrike"_s) {
        addNotation(UpDiagonalStrike);
    } else if (notation == "downdiagonalstrike"_s) {
        addNotation(DownDiagonalStrike);
    } else if (notation == "verticalstrike"_s) {
        addNotation(VerticalStrike);
    } else if (notation == "horizontalstrike"_s) {
        addNotation(HorizontalStrike);
    } else if (notation == "updiagonalarrow"_s) {
        addNotation(UpDiagonalArrow);
    } else if (notation == "phasorangle"_s) {
        addNotation(PhasorAngle);
    } else if (notation == "box"_s) {
        addNotation(Left);
        addNotation(Right);
        addNotation(Top);
        addNotation(Bottom);
    } else if (notation == "actuarial"_s) {
        addNotation(Right);
        addNotation(Top);
    } else if (notation == "madruwb"_s) {
        addNotation(Right);
        addNotation(Bottom);
    }
}

void MathMLMencloseElement::parseNotationAttribute()
{
    clearNotations();
    if (!hasAttribute(notationAttr)) {
        addNotation(LongDiv); // The default value is longdiv.
        return;
    }
    // We parse the list of whitespace-separated notation names.
    StringView value = attributeWithoutSynchronization(notationAttr).string();
    unsigned length = value.length();
    unsigned start = 0;
    while (start < length) {
        if (isASCIIWhitespace(value[start])) {
            start++;
            continue;
        }
        unsigned end = start + 1;
        while (end < length && !isASCIIWhitespace(value[end]))
            end++;
        addNotationFlags(value.substring(start, end - start));
        start = end;
    }
}

bool MathMLMencloseElement::hasNotation(MencloseNotationFlag notationFlag)
{
    if (!m_notationFlags)
        parseNotationAttribute();
    return m_notationFlags.value() & notationFlag;
}

void MathMLMencloseElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == notationAttr)
        m_notationFlags = std::nullopt;

    MathMLRowElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

}
#endif // ENABLE(MATHML)
