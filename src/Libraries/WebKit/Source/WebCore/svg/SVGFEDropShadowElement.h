/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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

#include "FEDropShadow.h"
#include <wtf/TZoneMalloc.h>
#include "SVGFilterPrimitiveStandardAttributes.h"

namespace WebCore {

class SVGFEDropShadowElement final : public SVGFilterPrimitiveStandardAttributes {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGFEDropShadowElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGFEDropShadowElement);
public:
    static Ref<SVGFEDropShadowElement> create(const QualifiedName&, Document&);
    
    void setStdDeviation(float stdDeviationX, float stdDeviationY);

    String in1() const { return m_in1->currentValue(); }
    float dx() const { return m_dx->currentValue(); }
    float dy() const { return m_dy->currentValue(); }
    float stdDeviationX() const { return m_stdDeviationX->currentValue(); }
    float stdDeviationY() const { return m_stdDeviationY->currentValue(); }

    SVGAnimatedString& in1Animated() { return m_in1; }
    SVGAnimatedNumber& dxAnimated() { return m_dx; }
    SVGAnimatedNumber& dyAnimated() { return m_dy; }
    SVGAnimatedNumber& stdDeviationXAnimated() { return m_stdDeviationX; }
    SVGAnimatedNumber& stdDeviationYAnimated() { return m_stdDeviationY; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGFEDropShadowElement, SVGFilterPrimitiveStandardAttributes>;

private:
    SVGFEDropShadowElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    bool setFilterEffectAttribute(FilterEffect&, const QualifiedName&) override;
    Vector<AtomString> filterEffectInputsNames() const override { return { AtomString { in1() } }; }
    bool isIdentity() const override;
    IntOutsets outsets(const FloatRect& targetBoundingBox, SVGUnitTypes::SVGUnitType primitiveUnits) const override;
    RefPtr<FilterEffect> createFilterEffect(const FilterEffectVector&, const GraphicsContext& destinationContext) const override;

    Ref<SVGAnimatedString> m_in1 { SVGAnimatedString::create(this) };
    Ref<SVGAnimatedNumber> m_dx { SVGAnimatedNumber::create(this, 2) };
    Ref<SVGAnimatedNumber> m_dy { SVGAnimatedNumber::create(this, 2) };
    Ref<SVGAnimatedNumber> m_stdDeviationX { SVGAnimatedNumber::create(this, 2) };
    Ref<SVGAnimatedNumber> m_stdDeviationY { SVGAnimatedNumber::create(this, 2) };
};
    
} // namespace WebCore
