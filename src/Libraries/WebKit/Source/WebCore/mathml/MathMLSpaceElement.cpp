/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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
#include "MathMLSpaceElement.h"

#if ENABLE(MATHML)

#include "NodeName.h"
#include "RenderMathMLSpace.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MathMLSpaceElement);

using namespace MathMLNames;

MathMLSpaceElement::MathMLSpaceElement(const QualifiedName& tagName, Document& document)
    : MathMLPresentationElement(tagName, document)
{
}

Ref<MathMLSpaceElement> MathMLSpaceElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new MathMLSpaceElement(tagName, document));
}

const MathMLElement::Length& MathMLSpaceElement::width()
{
    return cachedMathMLLength(MathMLNames::widthAttr, m_width);
}

const MathMLElement::Length& MathMLSpaceElement::height()
{
    return cachedMathMLLength(MathMLNames::heightAttr, m_height);
}

const MathMLElement::Length& MathMLSpaceElement::depth()
{
    return cachedMathMLLength(MathMLNames::depthAttr, m_depth);
}

void MathMLSpaceElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::widthAttr:
        m_width = std::nullopt;
        break;
    case AttributeNames::heightAttr:
        m_height = std::nullopt;
        break;
    case AttributeNames::depthAttr:
        m_depth = std::nullopt;
        break;
    default:
        break;
    }
    MathMLPresentationElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

RenderPtr<RenderElement> MathMLSpaceElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    ASSERT(hasTagName(MathMLNames::mspaceTag));
    return createRenderer<RenderMathMLSpace>(*this, WTFMove(style));
}

}

#endif // ENABLE(MATHML)
