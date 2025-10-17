/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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

#include "FETurbulence.h"
#include "SVGFilterPrimitiveStandardAttributes.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

enum SVGStitchOptions {
    SVG_STITCHTYPE_UNKNOWN  = 0,
    SVG_STITCHTYPE_STITCH   = 1,
    SVG_STITCHTYPE_NOSTITCH = 2
};

template<>
struct SVGPropertyTraits<SVGStitchOptions> {
    static unsigned highestEnumValue() { return SVG_STITCHTYPE_NOSTITCH; }

    static String toString(SVGStitchOptions type)
    {
        switch (type) {
        case SVG_STITCHTYPE_UNKNOWN:
            return emptyString();
        case SVG_STITCHTYPE_STITCH:
            return "stitch"_s;
        case SVG_STITCHTYPE_NOSTITCH:
            return "noStitch"_s;
        }

        ASSERT_NOT_REACHED();
        return emptyString();
    }

    static SVGStitchOptions fromString(const String& value)
    {
        if (value == "stitch"_s)
            return SVG_STITCHTYPE_STITCH;
        if (value == "noStitch"_s)
            return SVG_STITCHTYPE_NOSTITCH;
        return SVG_STITCHTYPE_UNKNOWN;
    }
};

template<>
struct SVGPropertyTraits<TurbulenceType> {
    static unsigned highestEnumValue() { return static_cast<unsigned>(TurbulenceType::Turbulence); }

    static String toString(TurbulenceType type)
    {
        switch (type) {
        case TurbulenceType::Unknown:
            return emptyString();
        case TurbulenceType::FractalNoise:
            return "fractalNoise"_s;
        case TurbulenceType::Turbulence:
            return "turbulence"_s;
        }

        ASSERT_NOT_REACHED();
        return emptyString();
    }

    static TurbulenceType fromString(const String& value)
    {
        if (value == "fractalNoise"_s)
            return TurbulenceType::FractalNoise;
        if (value == "turbulence"_s)
            return TurbulenceType::Turbulence;
        return TurbulenceType::Unknown;
    }
};

class SVGFETurbulenceElement final : public SVGFilterPrimitiveStandardAttributes {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGFETurbulenceElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGFETurbulenceElement);
public:
    static Ref<SVGFETurbulenceElement> create(const QualifiedName&, Document&);

    float baseFrequencyX() const { return m_baseFrequencyX->currentValue(); }
    float baseFrequencyY() const { return m_baseFrequencyY->currentValue(); }
    int numOctaves() const { return m_numOctaves->currentValue(); }
    float seed() const { return m_seed->currentValue(); }
    SVGStitchOptions stitchTiles() const { return m_stitchTiles->currentValue<SVGStitchOptions>(); }
    TurbulenceType type() const { return m_type->currentValue<TurbulenceType>(); }

    SVGAnimatedNumber& baseFrequencyXAnimated() { return m_baseFrequencyX; }
    SVGAnimatedNumber& baseFrequencyYAnimated() { return m_baseFrequencyY; }
    SVGAnimatedInteger& numOctavesAnimated() { return m_numOctaves; }
    SVGAnimatedNumber& seedAnimated() { return m_seed; }
    SVGAnimatedEnumeration& stitchTilesAnimated() { return m_stitchTiles; }
    SVGAnimatedEnumeration& typeAnimated() { return m_type; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGFETurbulenceElement, SVGFilterPrimitiveStandardAttributes>;

private:
    SVGFETurbulenceElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    bool setFilterEffectAttribute(FilterEffect&, const QualifiedName& attrName) override;
    RefPtr<FilterEffect> createFilterEffect(const FilterEffectVector&, const GraphicsContext& destinationContext) const override;

    Ref<SVGAnimatedNumber> m_baseFrequencyX { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedNumber> m_baseFrequencyY { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedInteger> m_numOctaves { SVGAnimatedInteger::create(this, 1) };
    Ref<SVGAnimatedNumber> m_seed { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedEnumeration> m_stitchTiles { SVGAnimatedEnumeration::create(this, SVG_STITCHTYPE_NOSTITCH) };
    Ref<SVGAnimatedEnumeration> m_type { SVGAnimatedEnumeration::create(this, TurbulenceType::Turbulence) };
};

} // namespace WebCore
