/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 5, 2024.
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

#include "SVGFELightElement.h"
#include "SVGFilterPrimitiveStandardAttributes.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FEDiffuseLighting;
class SVGColor;

class SVGFEDiffuseLightingElement final : public SVGFilterPrimitiveStandardAttributes {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGFEDiffuseLightingElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGFEDiffuseLightingElement);
public:
    static Ref<SVGFEDiffuseLightingElement> create(const QualifiedName&, Document&);
    void lightElementAttributeChanged(const SVGFELightElement*, const QualifiedName&);

    String in1() const { return m_in1->currentValue(); }
    float diffuseConstant() const { return m_diffuseConstant->currentValue(); }
    float surfaceScale() const { return m_surfaceScale->currentValue(); }
    float kernelUnitLengthX() const { return m_kernelUnitLengthX->currentValue(); }
    float kernelUnitLengthY() const { return m_kernelUnitLengthY->currentValue(); }

    SVGAnimatedString& in1Animated() { return m_in1; }
    SVGAnimatedNumber& diffuseConstantAnimated() { return m_diffuseConstant; }
    SVGAnimatedNumber& surfaceScaleAnimated() { return m_surfaceScale; }
    SVGAnimatedNumber& kernelUnitLengthXAnimated() { return m_kernelUnitLengthX; }
    SVGAnimatedNumber& kernelUnitLengthYAnimated() { return m_kernelUnitLengthY; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGFEDiffuseLightingElement, SVGFilterPrimitiveStandardAttributes>;

private:
    SVGFEDiffuseLightingElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    bool setFilterEffectAttribute(FilterEffect&, const QualifiedName&) override;
    Vector<AtomString> filterEffectInputsNames() const override { return { AtomString { in1() } }; }
    RefPtr<FilterEffect> createFilterEffect(const FilterEffectVector&, const GraphicsContext& destinationContext) const override;

    Ref<SVGAnimatedString> m_in1 { SVGAnimatedString::create(this) };
    Ref<SVGAnimatedNumber> m_diffuseConstant { SVGAnimatedNumber::create(this, 1) };
    Ref<SVGAnimatedNumber> m_surfaceScale { SVGAnimatedNumber::create(this, 1) };
    Ref<SVGAnimatedNumber> m_kernelUnitLengthX { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedNumber> m_kernelUnitLengthY { SVGAnimatedNumber::create(this) };
};

} // namespace WebCore
