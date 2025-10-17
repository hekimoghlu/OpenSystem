/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 17, 2025.
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
#include "FEMorphology.h"

#include "Filter.h"
#include "FEMorphologySoftwareApplier.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<FEMorphology> FEMorphology::create(MorphologyOperatorType type, float radiusX, float radiusY, DestinationColorSpace colorSpace)
{
    return adoptRef(*new FEMorphology(type, radiusX, radiusY, colorSpace));
}

FEMorphology::FEMorphology(MorphologyOperatorType type, float radiusX, float radiusY, DestinationColorSpace colorSpace)
    : FilterEffect(FilterEffect::Type::FEMorphology, colorSpace)
    , m_type(type)
    , m_radiusX(std::max(0.0f, radiusX))
    , m_radiusY(std::max(0.0f, radiusY))
{
}

bool FEMorphology::operator==(const FEMorphology& other) const
{
    return FilterEffect::operator==(other)
        && m_type == other.m_type
        && m_radiusX == other.m_radiusX
        && m_radiusY == other.m_radiusY;
}

bool FEMorphology::setMorphologyOperator(MorphologyOperatorType type)
{
    if (m_type == type)
        return false;
    m_type = type;
    return true;
}

bool FEMorphology::setRadiusX(float radiusX)
{
    radiusX = std::max(0.0f, radiusX);
    if (m_radiusX == radiusX)
        return false;
    m_radiusX = radiusX;
    return true;
}

bool FEMorphology::setRadiusY(float radiusY)
{
    radiusY = std::max(0.0f, radiusY);
    if (m_radiusY == radiusY)
        return false;
    m_radiusY = radiusY;
    return true;
}

FloatRect FEMorphology::calculateImageRect(const Filter& filter, std::span<const FloatRect> inputImageRects, const FloatRect& primitiveSubregion) const
{
    auto imageRect = inputImageRects[0];
    imageRect.inflate(filter.resolvedSize({ m_radiusX, m_radiusY }));
    return filter.clipToMaxEffectRect(imageRect, primitiveSubregion);
}

bool FEMorphology::resultIsAlphaImage(const FilterImageVector& inputs) const
{
    return inputs[0]->isAlphaImage();
}

std::unique_ptr<FilterEffectApplier> FEMorphology::createSoftwareApplier() const
{
    return FilterEffectApplier::create<FEMorphologySoftwareApplier>(*this);
}

static TextStream& operator<<(TextStream& ts, const MorphologyOperatorType& type)
{
    switch (type) {
    case MorphologyOperatorType::Unknown:
        ts << "UNKNOWN";
        break;
    case MorphologyOperatorType::Erode:
        ts << "ERODE";
        break;
    case MorphologyOperatorType::Dilate:
        ts << "DILATE";
        break;
    }
    return ts;
}

TextStream& FEMorphology::externalRepresentation(TextStream& ts, FilterRepresentation representation) const
{
    ts << indent << "[feMorphology";
    FilterEffect::externalRepresentation(ts, representation);

    ts << " operator=\"" << morphologyOperator() << "\"";
    ts << " radius=\"" << radiusX() << ", " << radiusY() << "\"";

    ts << "]\n";
    return ts;
}

} // namespace WebCore
