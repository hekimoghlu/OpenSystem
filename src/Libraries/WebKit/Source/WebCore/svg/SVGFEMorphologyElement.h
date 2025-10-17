/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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

#include "FEMorphology.h"
#include "SVGFilterPrimitiveStandardAttributes.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

template<>
struct SVGPropertyTraits<MorphologyOperatorType> {
    static unsigned highestEnumValue() { return static_cast<unsigned>(MorphologyOperatorType::Dilate); }

    static String toString(MorphologyOperatorType type)
    {
        switch (type) {
        case MorphologyOperatorType::Unknown:
            return emptyString();
        case MorphologyOperatorType::Erode:
            return "erode"_s;
        case MorphologyOperatorType::Dilate:
            return "dilate"_s;
        }

        ASSERT_NOT_REACHED();
        return emptyString();
    }

    static MorphologyOperatorType fromString(const String& value)
    {
        if (value == "erode"_s)
            return MorphologyOperatorType::Erode;
        if (value == "dilate"_s)
            return MorphologyOperatorType::Dilate;
        return MorphologyOperatorType::Unknown;
    }
};

class SVGFEMorphologyElement final : public SVGFilterPrimitiveStandardAttributes {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGFEMorphologyElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGFEMorphologyElement);
public:
    static Ref<SVGFEMorphologyElement> create(const QualifiedName&, Document&);

    String in1() const { return m_in1->currentValue(); }
    MorphologyOperatorType svgOperator() const { return m_svgOperator->currentValue<MorphologyOperatorType>(); }
    float radiusX() const { return m_radiusX->currentValue(); }
    float radiusY() const { return m_radiusY->currentValue(); }

    SVGAnimatedString& in1Animated() { return m_in1; }
    SVGAnimatedEnumeration& svgOperatorAnimated() { return m_svgOperator; }
    SVGAnimatedNumber& radiusXAnimated() { return m_radiusX; }
    SVGAnimatedNumber& radiusYAnimated() { return m_radiusY; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGFEMorphologyElement, SVGFilterPrimitiveStandardAttributes>;

private:
    SVGFEMorphologyElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    bool setFilterEffectAttribute(FilterEffect&, const QualifiedName&) override;
    Vector<AtomString> filterEffectInputsNames() const override { return { AtomString { in1() } }; }
    bool isIdentity() const override;
    IntOutsets outsets(const FloatRect& targetBoundingBox, SVGUnitTypes::SVGUnitType primitiveUnits) const override;
    RefPtr<FilterEffect> createFilterEffect(const FilterEffectVector&, const GraphicsContext& destinationContext) const override;

    Ref<SVGAnimatedString> m_in1 { SVGAnimatedString::create(this) };
    Ref<SVGAnimatedEnumeration> m_svgOperator { SVGAnimatedEnumeration::create(this, MorphologyOperatorType::Erode) };
    Ref<SVGAnimatedNumber> m_radiusX { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedNumber> m_radiusY { SVGAnimatedNumber::create(this) };
};

} // namespace WebCore
