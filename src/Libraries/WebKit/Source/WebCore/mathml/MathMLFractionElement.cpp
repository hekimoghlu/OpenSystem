/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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
#include "MathMLFractionElement.h"

#if ENABLE(MATHML)

#include "ElementInlines.h"
#include "NodeName.h"
#include "RenderMathMLFraction.h"
#include "Settings.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MathMLFractionElement);

using namespace MathMLNames;

inline MathMLFractionElement::MathMLFractionElement(const QualifiedName& tagName, Document& document)
    : MathMLRowElement(tagName, document)
{
}

Ref<MathMLFractionElement> MathMLFractionElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new MathMLFractionElement(tagName, document));
}

const MathMLElement::Length& MathMLFractionElement::lineThickness()
{
    if (m_lineThickness)
        return m_lineThickness.value();

    auto& thickness = attributeWithoutSynchronization(linethicknessAttr);
    if (document().settings().coreMathMLEnabled()) {
        m_lineThickness = parseMathMLLength(thickness, false);
        return m_lineThickness.value();
    }

    // The MathML3 recommendation states that "medium" is the default thickness.
    // However, it only states that "thin" and "thick" are respectively thiner and thicker.
    // The MathML in HTML5 implementation note suggests 50% and 200% and these values are also used in Gecko.
    m_lineThickness = Length();
    if (equalLettersIgnoringASCIICase(thickness, "thin"_s)) {
        m_lineThickness.value().type = LengthType::UnitLess;
        m_lineThickness.value().value = .5;
    } else if (equalLettersIgnoringASCIICase(thickness, "medium"_s)) {
        m_lineThickness.value().type = LengthType::UnitLess;
        m_lineThickness.value().value = 1;
    } else if (equalLettersIgnoringASCIICase(thickness, "thick"_s)) {
        m_lineThickness.value().type = LengthType::UnitLess;
        m_lineThickness.value().value = 2;
    } else
        m_lineThickness = parseMathMLLength(thickness, true);
    return m_lineThickness.value();
}

MathMLFractionElement::FractionAlignment MathMLFractionElement::cachedFractionAlignment(const QualifiedName& name, std::optional<FractionAlignment>& alignment)
{
    if (alignment)
        return alignment.value();

    if (document().settings().coreMathMLEnabled()) {
        alignment = FractionAlignmentCenter;
        return alignment.value();
    }

    auto& value = attributeWithoutSynchronization(name);
    if (equalLettersIgnoringASCIICase(value, "left"_s))
        alignment = FractionAlignmentLeft;
    else if (equalLettersIgnoringASCIICase(value, "right"_s))
        alignment = FractionAlignmentRight;
    else
        alignment = FractionAlignmentCenter;
    return alignment.value();
}

MathMLFractionElement::FractionAlignment MathMLFractionElement::numeratorAlignment()
{
    return cachedFractionAlignment(numalignAttr, m_numeratorAlignment);
}

MathMLFractionElement::FractionAlignment MathMLFractionElement::denominatorAlignment()
{
    return cachedFractionAlignment(denomalignAttr, m_denominatorAlignment);
}

void MathMLFractionElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::linethicknessAttr:
        m_lineThickness = std::nullopt;
        break;
    case AttributeNames::numalignAttr:
        m_numeratorAlignment = std::nullopt;
        break;
    case AttributeNames::denomalignAttr:
        m_denominatorAlignment = std::nullopt;
        break;
    default:
        break;
    }

    MathMLElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

RenderPtr<RenderElement> MathMLFractionElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    ASSERT(hasTagName(MathMLNames::mfracTag));
    return createRenderer<RenderMathMLFraction>(*this, WTFMove(style));
}

}

#endif // ENABLE(MATHML)
