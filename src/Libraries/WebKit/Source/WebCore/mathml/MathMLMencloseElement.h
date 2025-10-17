/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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

#include "Element.h"
#include "MathMLRowElement.h"

namespace WebCore {

class MathMLMencloseElement final: public MathMLRowElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MathMLMencloseElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MathMLMencloseElement);
public:
    static Ref<MathMLMencloseElement> create(const QualifiedName& tagName, Document&);

    enum MencloseNotationFlag {
        LongDiv = 1 << 1,
        RoundedBox = 1 << 2,
        Circle = 1 << 3,
        Left = 1 << 4,
        Right = 1 << 5,
        Top = 1 << 6,
        Bottom = 1 << 7,
        UpDiagonalStrike = 1 << 8,
        DownDiagonalStrike = 1 << 9,
        VerticalStrike = 1 << 10,
        HorizontalStrike = 1 << 11,
        UpDiagonalArrow = 1 << 12, // FIXME: updiagonalarrow is not implemented. See http://wkb.ug/127466
        PhasorAngle = 1 << 13 // FIXME: phasorangle is not implemented. See http://wkb.ug/127466
        // We do not implement the Radical notation. Authors should instead use the <msqrt> element.
    };
    bool hasNotation(MencloseNotationFlag);

private:
    MathMLMencloseElement(const QualifiedName&, Document&);
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void parseNotationAttribute();
    void clearNotations() { m_notationFlags = 0; }
    void addNotation(MencloseNotationFlag notationFlag) { m_notationFlags.value() |= notationFlag; }
    void addNotationFlags(StringView notation);
    std::optional<uint16_t> m_notationFlags;
};

}

#endif // ENABLE(MATHML)
