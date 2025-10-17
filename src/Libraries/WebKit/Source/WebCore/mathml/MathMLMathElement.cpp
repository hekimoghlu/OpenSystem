/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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
#include "MathMLMathElement.h"

#if ENABLE(MATHML)

#include "MathMLNames.h"
#include "RenderMathMLMath.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MathMLMathElement);

using namespace MathMLNames;

inline MathMLMathElement::MathMLMathElement(const QualifiedName& tagName, Document& document)
    : MathMLRowElement(tagName, document, TypeFlag::HasCustomStyleResolveCallbacks)
{
}

Ref<MathMLMathElement> MathMLMathElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new MathMLMathElement(tagName, document));
}

RenderPtr<RenderElement> MathMLMathElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    return createRenderer<RenderMathMLMath>(*this, WTFMove(style));
}

void MathMLMathElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == mathvariantAttr) {
        m_mathVariant = std::nullopt;
        if (renderer())
            MathMLStyle::resolveMathMLStyleTree(renderer());
    }

    MathMLElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void MathMLMathElement::didAttachRenderers()
{
    MathMLRowElement::didAttachRenderers();

    MathMLStyle::resolveMathMLStyleTree(renderer());
}

}

#endif // ENABLE(MATHML)
