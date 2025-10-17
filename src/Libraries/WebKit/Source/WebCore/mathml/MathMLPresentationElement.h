/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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

#include "MathMLElement.h"

namespace WebCore {

class MathMLPresentationElement : public MathMLElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MathMLPresentationElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MathMLPresentationElement);
public:
    static Ref<MathMLPresentationElement> create(const QualifiedName& tagName, Document&);

protected:
    MathMLPresentationElement(const QualifiedName& tagName, Document&, OptionSet<TypeFlag> = { });
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;

    static std::optional<bool> toOptionalBool(const BooleanValue& value) { return value == BooleanValue::Default ? std::nullopt : std::optional<bool>(value == BooleanValue::True); }
    const BooleanValue& cachedBooleanAttribute(const QualifiedName&, std::optional<BooleanValue>&);

    static Length parseMathMLLength(const String&, bool acceptLegacyMathMLLengths);
    const Length& cachedMathMLLength(const QualifiedName&, std::optional<Length>&);

    virtual bool acceptsMathVariantAttribute() { return false; }
    std::optional<MathVariant> specifiedMathVariant() final;

    std::optional<MathVariant> m_mathVariant;

private:
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;
    bool isPresentationMathML() const final { return true; }

    static Length parseNumberAndUnit(StringView, bool acceptLegacyMathMLLengths);
    static Length parseNamedSpace(StringView);
    static MathVariant parseMathVariantAttribute(const AtomString& attributeValue);
};

}

#endif // ENABLE(MATHML)
