/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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

#include "ColorComponents.h"
#include "FilterEffect.h"

namespace WebCore {

enum class TurbulenceType : uint8_t {
    Unknown,
    FractalNoise,
    Turbulence
};

class FETurbulence : public FilterEffect {
public:
    WEBCORE_EXPORT static Ref<FETurbulence> create(TurbulenceType, float baseFrequencyX, float baseFrequencyY, int numOctaves, float seed, bool stitchTiles, DestinationColorSpace = DestinationColorSpace::SRGB());

    bool operator==(const FETurbulence&) const;

    TurbulenceType type() const { return m_type; }
    bool setType(TurbulenceType);

    float baseFrequencyX() const { return m_baseFrequencyX; }
    bool setBaseFrequencyX(float);

    float baseFrequencyY() const { return m_baseFrequencyY; }
    bool setBaseFrequencyY(float);

    float seed() const { return m_seed; }
    bool setSeed(float);

    int numOctaves() const { return m_numOctaves; }
    bool setNumOctaves(int);

    bool stitchTiles() const { return m_stitchTiles; }
    bool setStitchTiles(bool);

private:
    FETurbulence(TurbulenceType, float baseFrequencyX, float baseFrequencyY, int numOctaves, float seed, bool stitchTiles, DestinationColorSpace);

    bool operator==(const FilterEffect& other) const override { return areEqual<FETurbulence>(*this, other); }

    unsigned numberOfEffectInputs() const override { return 0; }

    FloatRect calculateImageRect(const Filter&, std::span<const FloatRect> inputImageRects, const FloatRect& primitiveSubregion) const override;

    std::unique_ptr<FilterEffectApplier> createSoftwareApplier() const override;

    WTF::TextStream& externalRepresentation(WTF::TextStream&, FilterRepresentation) const override;

    TurbulenceType m_type;
    float m_baseFrequencyX;
    float m_baseFrequencyY;
    int m_numOctaves;
    float m_seed;
    bool m_stitchTiles;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_FILTER_FUNCTION(FETurbulence)
