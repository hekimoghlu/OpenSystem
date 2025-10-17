/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 2, 2024.
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
#include "RenderMathMLFencedOperator.h"

#if ENABLE(MATHML)

#include "MathMLOperatorDictionary.h"
#include "MathMLOperatorElement.h"
#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderStyleInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace MathMLOperatorDictionary;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderMathMLFencedOperator);

RenderMathMLFencedOperator::RenderMathMLFencedOperator(Document& document, RenderStyle&& style, const String& operatorString, MathMLOperatorDictionary::Form form, unsigned short flags)
    : RenderMathMLOperator(Type::MathMLFencedOperator, document, WTFMove(style))
    , m_operatorForm(form)
    , m_operatorFlags(flags)
{
    updateOperatorContent(operatorString);
    ASSERT(isRenderMathMLFencedOperator());
}

RenderMathMLFencedOperator::~RenderMathMLFencedOperator() = default;

void RenderMathMLFencedOperator::updateOperatorContent(const String& operatorString)
{
    m_operatorChar = MathMLOperatorElement::parseOperatorChar(operatorString);

    // We try and read spacing and boolean properties from the operator dictionary.
    // However we preserve the Fence and Separator properties specified in the constructor.
    if (auto entry = search(m_operatorChar.character, m_operatorForm, true)) {
        m_leadingSpaceInMathUnit = entry.value().leadingSpaceInMathUnit;
        m_trailingSpaceInMathUnit = entry.value().trailingSpaceInMathUnit;
        m_operatorFlags = (m_operatorFlags & (MathMLOperatorDictionary::Fence | MathMLOperatorDictionary::Separator)) | entry.value().flags;
    } else {
        m_operatorFlags &= MathMLOperatorDictionary::Fence | MathMLOperatorDictionary::Separator; // Flags are disabled by default.
        m_leadingSpaceInMathUnit = 5; // Default spacing is thickmathspace.
        m_trailingSpaceInMathUnit = 5; // Default spacing is thickmathspace.
    }

    updateMathOperator();
}

LayoutUnit RenderMathMLFencedOperator::leadingSpace() const
{
    MathMLElement::Length leadingSpace;
    leadingSpace.type = MathMLElement::LengthType::MathUnit;
    leadingSpace.value = static_cast<float>(m_leadingSpaceInMathUnit);
    return toUserUnits(leadingSpace, style(), 0);
}

LayoutUnit RenderMathMLFencedOperator::trailingSpace() const
{
    MathMLElement::Length trailingSpace;
    trailingSpace.type = MathMLElement::LengthType::MathUnit;
    trailingSpace.value = static_cast<float>(m_trailingSpaceInMathUnit);
    return toUserUnits(trailingSpace, style(), 0);
}

}

#endif
