/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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
#include "config.h"
#include "FETurbulence.h"

#include "FETurbulenceSoftwareApplier.h"
#include "Filter.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<FETurbulence> FETurbulence::create(TurbulenceType type, float baseFrequencyX, float baseFrequencyY, int numOctaves, float seed, bool stitchTiles, DestinationColorSpace colorSpace)
{
    return adoptRef(*new FETurbulence(type, baseFrequencyX, baseFrequencyY, numOctaves, seed, stitchTiles, colorSpace));
}

FETurbulence::FETurbulence(TurbulenceType type, float baseFrequencyX, float baseFrequencyY, int numOctaves, float seed, bool stitchTiles, DestinationColorSpace colorSpace)
    : FilterEffect(FilterEffect::Type::FETurbulence, colorSpace)
    , m_type(type)
    , m_baseFrequencyX(baseFrequencyX)
    , m_baseFrequencyY(baseFrequencyY)
    , m_numOctaves(numOctaves)
    , m_seed(seed)
    , m_stitchTiles(stitchTiles)
{
}

bool FETurbulence::operator==(const FETurbulence& other) const
{
    return FilterEffect::operator==(other)
        && m_type == other.m_type
        && m_baseFrequencyX == other.m_baseFrequencyX
        && m_baseFrequencyY == other.m_baseFrequencyY
        && m_numOctaves == other.m_numOctaves
        && m_seed == other.m_seed
        && m_stitchTiles == other.m_stitchTiles;
}

bool FETurbulence::setType(TurbulenceType type)
{
    if (m_type == type)
        return false;
    m_type = type;
    return true;
}

bool FETurbulence::setBaseFrequencyY(float baseFrequencyY)
{
    if (m_baseFrequencyY == baseFrequencyY)
        return false;
    m_baseFrequencyY = baseFrequencyY;
    return true;
}

bool FETurbulence::setBaseFrequencyX(float baseFrequencyX)
{
    if (m_baseFrequencyX == baseFrequencyX)
        return false;
    m_baseFrequencyX = baseFrequencyX;
    return true;
}

bool FETurbulence::setSeed(float seed)
{
    if (m_seed == seed)
        return false;
    m_seed = seed;
    return true;
}

bool FETurbulence::setNumOctaves(int numOctaves)
{
    if (m_numOctaves == numOctaves)
        return false;
    m_numOctaves = numOctaves;
    return true;
}

bool FETurbulence::setStitchTiles(bool stitch)
{
    if (m_stitchTiles == stitch)
        return false;
    m_stitchTiles = stitch;
    return true;
}

FloatRect FETurbulence::calculateImageRect(const Filter& filter, std::span<const FloatRect>, const FloatRect& primitiveSubregion) const
{
    return filter.maxEffectRect(primitiveSubregion);
}

std::unique_ptr<FilterEffectApplier> FETurbulence::createSoftwareApplier() const
{
    return FilterEffectApplier::create<FETurbulenceSoftwareApplier>(*this);
}

static TextStream& operator<<(TextStream& ts, TurbulenceType type)
{
    switch (type) {
    case TurbulenceType::Unknown:
        ts << "UNKNOWN";
        break;
    case TurbulenceType::Turbulence:
        ts << "TURBULENCE";
        break;
    case TurbulenceType::FractalNoise:
        ts << "NOISE";
        break;
    }
    return ts;
}

TextStream& FETurbulence::externalRepresentation(TextStream& ts, FilterRepresentation representation) const
{
    ts << indent << "[feTurbulence";
    FilterEffect::externalRepresentation(ts, representation);
    
    ts << " type=\"" << type() << "\"";
    ts << " baseFrequency=\"" << baseFrequencyX() << ", " << baseFrequencyY() << "\"";
    ts << " seed=\"" << seed() << "\"";
    ts << " numOctaves=\"" << numOctaves() << "\"";
    ts << " stitchTiles=\"" << stitchTiles() << "\"";

    ts << "]\n";
    return ts;
}

} // namespace WebCore
