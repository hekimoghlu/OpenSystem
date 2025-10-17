/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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

#include "FEGaussianBlur.h"
#include "SVGFEConvolveMatrixElement.h"
#include "SVGFilterPrimitiveStandardAttributes.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGFEGaussianBlurElement final : public SVGFilterPrimitiveStandardAttributes {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGFEGaussianBlurElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGFEGaussianBlurElement);
public:
    static Ref<SVGFEGaussianBlurElement> create(const QualifiedName&, Document&);

    void setStdDeviation(float stdDeviationX, float stdDeviationY);

    String in1() const { return m_in1->currentValue(); }
    float stdDeviationX() const { return m_stdDeviationX->currentValue(); }
    float stdDeviationY() const { return m_stdDeviationY->currentValue(); }
    EdgeModeType edgeMode() const { return m_edgeMode->currentValue<EdgeModeType>(); }

    SVGAnimatedString& in1Animated() { return m_in1; }
    SVGAnimatedNumber& stdDeviationXAnimated() { return m_stdDeviationX; }
    SVGAnimatedNumber& stdDeviationYAnimated() { return m_stdDeviationY; }
    SVGAnimatedEnumeration& edgeModeAnimated() { return m_edgeMode; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGFEGaussianBlurElement, SVGFilterPrimitiveStandardAttributes>;

private:
    SVGFEGaussianBlurElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    bool setFilterEffectAttribute(FilterEffect&, const QualifiedName&) override;
    Vector<AtomString> filterEffectInputsNames() const override { return { AtomString { in1() } }; }
    bool isIdentity() const override;
    IntOutsets outsets(const FloatRect& targetBoundingBox, SVGUnitTypes::SVGUnitType primitiveUnits) const override;
    RefPtr<FilterEffect> createFilterEffect(const FilterEffectVector&, const GraphicsContext& destinationContext) const override;

    Ref<SVGAnimatedString> m_in1 { SVGAnimatedString::create(this) };
    Ref<SVGAnimatedNumber> m_stdDeviationX { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedNumber> m_stdDeviationY { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedEnumeration> m_edgeMode { SVGAnimatedEnumeration::create(this, EdgeModeType::None) };
};

} // namespace WebCore
