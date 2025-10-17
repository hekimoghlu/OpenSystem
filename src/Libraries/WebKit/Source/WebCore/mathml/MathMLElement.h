/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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

#include "MathMLNames.h"
#include "StyledElement.h"

namespace WebCore {

class MathMLElement : public StyledElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MathMLElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MathMLElement);
public:
    static Ref<MathMLElement> create(const QualifiedName& tagName, Document&);

    unsigned colSpan() const;
    unsigned rowSpan() const;

    virtual bool isMathMLToken() const { return false; }
    virtual bool isSemanticAnnotation() const { return false; }
    virtual bool isPresentationMathML() const { return false; }

    bool hasTagName(const MathMLQualifiedName& name) const { return hasLocalName(name.localName()); }

    // MathML lengths (https://www.w3.org/TR/MathML3/chapter2.html#fund.units)
    // TeX's Math Unit is used internally for named spaces (1 mu = 1/18 em).
    // Unitless values are interpreted as a multiple of a reference value.
    enum class LengthType { Cm, Em, Ex, In, MathUnit, Mm, ParsingFailed, Pc, Percentage, Pt, Px, UnitLess };
    struct Length {
        LengthType type { LengthType::ParsingFailed };
        float value { 0 };
    };

    enum class BooleanValue { True, False, Default };

    // These are the mathvariant values from the MathML recommendation.
    // The special value none means that no explicit mathvariant value has been specified.
    // Note that the numeral values are important for the computation performed in the mathVariant function of RenderMathMLToken, do not change them!
    enum class MathVariant {
        None = 0,
        Normal = 1,
        Bold = 2,
        Italic = 3,
        BoldItalic = 4,
        Script = 5,
        BoldScript = 6,
        Fraktur = 7,
        DoubleStruck = 8,
        BoldFraktur = 9,
        SansSerif = 10,
        BoldSansSerif = 11,
        SansSerifItalic = 12,
        SansSerifBoldItalic = 13,
        Monospace = 14,
        Initial = 15,
        Tailed = 16,
        Looped = 17,
        Stretched = 18
    };

    virtual std::optional<MathVariant> specifiedMathVariant() { return std::nullopt; }

    virtual void updateSelectedChild() { }

protected:
    MathMLElement(const QualifiedName& tagName, Document&, OptionSet<TypeFlag> = { });

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    bool childShouldCreateRenderer(const Node&) const override;

    bool hasPresentationalHintsForAttribute(const QualifiedName&) const override;
    void collectPresentationalHintsForAttribute(const QualifiedName&, const AtomString&, MutableStyleProperties&) override;

    bool willRespondToMouseClickEventsWithEditability(Editability) const override;
    void defaultEventHandler(Event&) override;

private:
    bool canStartSelection() const final;
    bool isKeyboardFocusable(KeyboardEvent*) const final;
    bool isMouseFocusable() const final;
    bool isURLAttribute(const Attribute&) const final;
    bool supportsFocus() const final;
};

inline bool Node::hasTagName(const MathMLQualifiedName& name) const
{
    return isMathMLElement() && downcast<MathMLElement>(*this).hasTagName(name);
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::MathMLElement)
    static bool isType(const WebCore::Node& node) { return node.isMathMLElement(); }
SPECIALIZE_TYPE_TRAITS_END()

#include "MathMLElementTypeHelpers.h"

#endif // ENABLE(MATHML)
