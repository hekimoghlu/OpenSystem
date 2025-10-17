/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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

#include "FilterEffect.h"

namespace WebCore {

enum class MorphologyOperatorType : uint8_t {
    Unknown,
    Erode,
    Dilate
};

class FEMorphology : public FilterEffect {
public:
    WEBCORE_EXPORT static Ref<FEMorphology> create(MorphologyOperatorType, float radiusX, float radiusY, DestinationColorSpace = DestinationColorSpace::SRGB());

    bool operator==(const FEMorphology&) const;

    MorphologyOperatorType morphologyOperator() const { return m_type; }
    bool setMorphologyOperator(MorphologyOperatorType);

    float radiusX() const { return m_radiusX; }
    bool setRadiusX(float);

    float radiusY() const { return m_radiusY; }
    bool setRadiusY(float);

private:
    FEMorphology(MorphologyOperatorType, float radiusX, float radiusY, DestinationColorSpace);

    bool operator==(const FilterEffect& other) const override { return areEqual<FEMorphology>(*this, other); }

    FloatRect calculateImageRect(const Filter&, std::span<const FloatRect> inputImageRects, const FloatRect& primitiveSubregion) const override;

    bool resultIsAlphaImage(const FilterImageVector& inputs) const override;

    std::unique_ptr<FilterEffectApplier> createSoftwareApplier() const override;

    WTF::TextStream& externalRepresentation(WTF::TextStream&, FilterRepresentation) const override;

    MorphologyOperatorType m_type;
    float m_radiusX;
    float m_radiusY;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_FILTER_FUNCTION(FEMorphology)
