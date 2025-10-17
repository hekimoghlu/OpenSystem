/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 22, 2021.
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
#include "MathMLPaddedElement.h"

#if ENABLE(MATHML)

#include "NodeName.h"
#include "RenderMathMLPadded.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MathMLPaddedElement);

using namespace MathMLNames;

inline MathMLPaddedElement::MathMLPaddedElement(const QualifiedName& tagName, Document& document)
    : MathMLRowElement(tagName, document)
{
}

Ref<MathMLPaddedElement> MathMLPaddedElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new MathMLPaddedElement(tagName, document));
}

const MathMLElement::Length& MathMLPaddedElement::width()
{
    return cachedMathMLLength(MathMLNames::widthAttr, m_width);
}

const MathMLElement::Length& MathMLPaddedElement::height()
{
    return cachedMathMLLength(MathMLNames::heightAttr, m_height);
}

const MathMLElement::Length& MathMLPaddedElement::depth()
{
    return cachedMathMLLength(MathMLNames::depthAttr, m_depth);
}

const MathMLElement::Length& MathMLPaddedElement::lspace()
{
    return cachedMathMLLength(MathMLNames::lspaceAttr, m_lspace);
}

const MathMLElement::Length& MathMLPaddedElement::voffset()
{
    return cachedMathMLLength(MathMLNames::voffsetAttr, m_voffset);
}

void MathMLPaddedElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
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
    case AttributeNames::lspaceAttr:
        m_lspace = std::nullopt;
        break;
    case AttributeNames::voffsetAttr:
        m_voffset = std::nullopt;
        break;
    default:
        break;
    }
    MathMLElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

RenderPtr<RenderElement> MathMLPaddedElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    ASSERT(hasTagName(MathMLNames::mpaddedTag));
    return createRenderer<RenderMathMLPadded>(*this, WTFMove(style));
}

}

#endif // ENABLE(MATHML)
