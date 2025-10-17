/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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
#pragma once

#if ENABLE(MATHML)

#include "MathMLOperatorDictionary.h"
#include "MathMLOperatorElement.h"
#include "RenderMathMLOperator.h"

namespace WebCore {

class RenderMathMLFencedOperator final : public RenderMathMLOperator {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderMathMLFencedOperator);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderMathMLFencedOperator);
public:
    RenderMathMLFencedOperator(Document&, RenderStyle&&, const String& operatorString, MathMLOperatorDictionary::Form, unsigned short flags = 0);
    virtual ~RenderMathMLFencedOperator();

    void updateOperatorContent(const String&);

private:
    bool isVertical() const final { return m_operatorChar.isVertical; }
    char32_t textContent() const final { return m_operatorChar.character; }
    LayoutUnit leadingSpace() const final;
    LayoutUnit trailingSpace() const final;

    // minsize always has the default value "1em".
    LayoutUnit minSize() const final { return LayoutUnit(style().fontCascade().size()); }

    // maxsize always has the default value "infinity".
    LayoutUnit maxSize() const final { return intMaxForLayoutUnit; }

    bool hasOperatorFlag(MathMLOperatorDictionary::Flag flag) const final { return m_operatorFlags & flag; }

    // We always use the MathOperator class for anonymous mfenced operators, since they do not have text content in the DOM.
    bool useMathOperator() const final { return true; }

    MathMLOperatorElement::OperatorChar m_operatorChar;
    unsigned short m_leadingSpaceInMathUnit;
    unsigned short m_trailingSpaceInMathUnit;
    MathMLOperatorDictionary::Form m_operatorForm;
    unsigned short m_operatorFlags;
};

}; // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderMathMLFencedOperator, isRenderMathMLFencedOperator())

#endif // ENABLE(MATHML)
