/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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
#include "MathMLUnderOverElement.h"

#if ENABLE(MATHML)

#include "RenderMathMLUnderOver.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MathMLUnderOverElement);

using namespace MathMLNames;

inline MathMLUnderOverElement::MathMLUnderOverElement(const QualifiedName& tagName, Document& document)
    : MathMLScriptsElement(tagName, document)
{
}

Ref<MathMLUnderOverElement> MathMLUnderOverElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new MathMLUnderOverElement(tagName, document));
}

const MathMLElement::BooleanValue& MathMLUnderOverElement::accent()
{
    return cachedBooleanAttribute(accentAttr, m_accent);
}

const MathMLElement::BooleanValue& MathMLUnderOverElement::accentUnder()
{
    return cachedBooleanAttribute(accentunderAttr, m_accentUnder);
}

void MathMLUnderOverElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == accentAttr)
        m_accent = std::nullopt;
    else if (name == accentunderAttr)
        m_accentUnder = std::nullopt;

    MathMLElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

RenderPtr<RenderElement> MathMLUnderOverElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    ASSERT(hasTagName(munderTag) || hasTagName(moverTag) || hasTagName(munderoverTag));
    return createRenderer<RenderMathMLUnderOver>(*this, WTFMove(style));
}

}

#endif // ENABLE(MATHML)
