/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 1, 2023.
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

#include "CommonAtomStrings.h"
#include "FEConvolveMatrix.h"
#include "SVGFilterPrimitiveStandardAttributes.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

template<>
struct SVGPropertyTraits<EdgeModeType> {
    static unsigned highestEnumValue() { return static_cast<unsigned>(EdgeModeType::None); }
    static EdgeModeType initialValue() { return EdgeModeType::None; }

    static String toString(EdgeModeType type)
    {
        switch (type) {
        case EdgeModeType::Unknown:
            return emptyString();
        case EdgeModeType::Duplicate:
            return "duplicate"_s;
        case EdgeModeType::Wrap:
            return "wrap"_s;
        case EdgeModeType::None:
            return noneAtom();
        }

        ASSERT_NOT_REACHED();
        return emptyString();
    }

    static EdgeModeType fromString(const String& value)
    {
        if (value == "duplicate"_s)
            return EdgeModeType::Duplicate;
        if (value == "wrap"_s)
            return EdgeModeType::Wrap;
        if (value == noneAtom())
            return EdgeModeType::None;
        return EdgeModeType::Unknown;
    }
};

class SVGFEConvolveMatrixElement final : public SVGFilterPrimitiveStandardAttributes {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGFEConvolveMatrixElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGFEConvolveMatrixElement);
public:
    static Ref<SVGFEConvolveMatrixElement> create(const QualifiedName&, Document&);

    void setOrder(float orderX, float orderY);
    void setKernelUnitLength(float kernelUnitLengthX, float kernelUnitLengthY);

    String in1() const { return m_in1->currentValue(); }
    int orderX() const { return m_orderX->currentValue(); }
    int orderY() const { return m_orderY->currentValue(); }
    const SVGNumberList& kernelMatrix() const { return m_kernelMatrix->currentValue(); }
    float divisor() const { return m_divisor->currentValue(); }
    float bias() const { return m_bias->currentValue(); }
    int targetX() const { return m_targetX->currentValue(); }
    int targetY() const { return m_targetY->currentValue(); }
    EdgeModeType edgeMode() const { return m_edgeMode->currentValue<EdgeModeType>(); }
    float kernelUnitLengthX() const { return m_kernelUnitLengthX->currentValue(); }
    float kernelUnitLengthY() const { return m_kernelUnitLengthY->currentValue(); }
    bool preserveAlpha() const { return m_preserveAlpha->currentValue(); }

    SVGAnimatedString& in1Animated() { return m_in1; }
    SVGAnimatedInteger& orderXAnimated() { return m_orderX; }
    SVGAnimatedInteger& orderYAnimated() { return m_orderY; }
    SVGAnimatedNumberList& kernelMatrixAnimated() { return m_kernelMatrix; }
    SVGAnimatedNumber& divisorAnimated() { return m_divisor; }
    SVGAnimatedNumber& biasAnimated() { return m_bias; }
    SVGAnimatedInteger& targetXAnimated() { return m_targetX; }
    SVGAnimatedInteger& targetYAnimated() { return m_targetY; }
    SVGAnimatedEnumeration& edgeModeAnimated() { return m_edgeMode; }
    SVGAnimatedNumber& kernelUnitLengthXAnimated() { return m_kernelUnitLengthX; }
    SVGAnimatedNumber& kernelUnitLengthYAnimated() { return m_kernelUnitLengthY; }
    SVGAnimatedBoolean& preserveAlphaAnimated() { return m_preserveAlpha; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGFEConvolveMatrixElement, SVGFilterPrimitiveStandardAttributes>;

private:
    SVGFEConvolveMatrixElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    bool isValidTargetXOffset() const;
    bool isValidTargetYOffset() const;
    bool setFilterEffectAttribute(FilterEffect&, const QualifiedName&) override;
    Vector<AtomString> filterEffectInputsNames() const override { return { AtomString { in1() } }; }
    RefPtr<FilterEffect> createFilterEffect(const FilterEffectVector&, const GraphicsContext& destinationContext) const override;

    Ref<SVGAnimatedString> m_in1 { SVGAnimatedString::create(this) };
    Ref<SVGAnimatedInteger> m_orderX { SVGAnimatedInteger::create(this) };
    Ref<SVGAnimatedInteger> m_orderY { SVGAnimatedInteger::create(this) };
    Ref<SVGAnimatedNumberList> m_kernelMatrix { SVGAnimatedNumberList::create(this) };
    Ref<SVGAnimatedNumber> m_divisor { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedNumber> m_bias { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedInteger> m_targetX { SVGAnimatedInteger::create(this) };
    Ref<SVGAnimatedInteger> m_targetY { SVGAnimatedInteger::create(this) };
    Ref<SVGAnimatedEnumeration> m_edgeMode { SVGAnimatedEnumeration::create(this, EdgeModeType::Duplicate) };
    Ref<SVGAnimatedNumber> m_kernelUnitLengthX { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedNumber> m_kernelUnitLengthY { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedBoolean> m_preserveAlpha { SVGAnimatedBoolean::create(this) };
};

} // namespace WebCore
