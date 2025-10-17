/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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

#include "FEColorMatrix.h"
#include "SVGFilterPrimitiveStandardAttributes.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

template<>
struct SVGPropertyTraits<ColorMatrixType> {
    static unsigned highestEnumValue() { return enumToUnderlyingType(ColorMatrixType::FECOLORMATRIX_TYPE_LUMINANCETOALPHA); }

    static String toString(ColorMatrixType type)
    {
        switch (type) {
        case ColorMatrixType::FECOLORMATRIX_TYPE_UNKNOWN:
            return emptyString();
        case ColorMatrixType::FECOLORMATRIX_TYPE_MATRIX:
            return "matrix"_s;
        case ColorMatrixType::FECOLORMATRIX_TYPE_SATURATE:
            return "saturate"_s;
        case ColorMatrixType::FECOLORMATRIX_TYPE_HUEROTATE:
            return "hueRotate"_s;
        case ColorMatrixType::FECOLORMATRIX_TYPE_LUMINANCETOALPHA:
            return "luminanceToAlpha"_s;
        }

        ASSERT_NOT_REACHED();
        return emptyString();
    }

    static ColorMatrixType fromString(const String& value)
    {
        if (value == "matrix"_s)
            return ColorMatrixType::FECOLORMATRIX_TYPE_MATRIX;
        if (value == "saturate"_s)
            return ColorMatrixType::FECOLORMATRIX_TYPE_SATURATE;
        if (value == "hueRotate"_s)
            return ColorMatrixType::FECOLORMATRIX_TYPE_HUEROTATE;
        if (value == "luminanceToAlpha"_s)
            return ColorMatrixType::FECOLORMATRIX_TYPE_LUMINANCETOALPHA;
        return ColorMatrixType::FECOLORMATRIX_TYPE_UNKNOWN;
    }
};

class SVGFEColorMatrixElement final : public SVGFilterPrimitiveStandardAttributes {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGFEColorMatrixElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGFEColorMatrixElement);
public:
    static Ref<SVGFEColorMatrixElement> create(const QualifiedName&, Document&);

    String in1() const { return m_in1->currentValue(); }
    ColorMatrixType type() const { return m_type->currentValue<ColorMatrixType>(); }
    const SVGNumberList& values() const { return m_values->currentValue(); }

    SVGAnimatedString& in1Animated() { return m_in1; }
    SVGAnimatedEnumeration& typeAnimated() { return m_type; }
    SVGAnimatedNumberList& valuesAnimated() { return m_values; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGFEColorMatrixElement, SVGFilterPrimitiveStandardAttributes>;

private:
    SVGFEColorMatrixElement(const QualifiedName&, Document&);

    bool isInvalidValuesLength() const;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    bool setFilterEffectAttribute(FilterEffect&, const QualifiedName&) override;
    Vector<AtomString> filterEffectInputsNames() const override { return { AtomString { in1() } }; }
    RefPtr<FilterEffect> createFilterEffect(const FilterEffectVector&, const GraphicsContext& destinationContext) const override;

    Ref<SVGAnimatedString> m_in1 { SVGAnimatedString::create(this) };
    Ref<SVGAnimatedEnumeration> m_type { SVGAnimatedEnumeration::create(this, ColorMatrixType::FECOLORMATRIX_TYPE_MATRIX) };
    Ref<SVGAnimatedNumberList> m_values { SVGAnimatedNumberList::create(this) };
};

} // namespace WebCore
