/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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
#include "PointLightSource.h"

#include "Filter.h"
#include "FilterImage.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<PointLightSource> PointLightSource::create(const FloatPoint3D& position)
{
    return adoptRef(*new PointLightSource(position));
}

PointLightSource::PointLightSource(const FloatPoint3D& position)
    : LightSource(LightType::LS_POINT)
    , m_position(position)
{
}

bool PointLightSource::operator==(const PointLightSource& other) const
{
    return LightSource::operator==(other) && m_position == other.m_position;
}

void PointLightSource::initPaintingData(const Filter& filter, const FilterImage& result, PaintingData&) const
{
    auto position = filter.resolvedPoint3D(m_position);
    auto absolutePosition = filter.scaledByFilterScale(position.xy());
    m_bufferPosition.setXY(result.mappedAbsolutePoint(absolutePosition));

    // To scale Z, map a point offset from position in the x direction by z.
    auto absoluteMappedZ = filter.scaledByFilterScale(FloatPoint { position.x() + position.z(), position.y() });
    m_bufferPosition.setZ(result.mappedAbsolutePoint(absoluteMappedZ).x() - m_bufferPosition.x());
}

LightSource::ComputedLightingData PointLightSource::computePixelLightingData(const PaintingData& paintingData, int x, int y, float z) const
{
    FloatPoint3D lightVector = {
        m_bufferPosition.x() - x,
        m_bufferPosition.y() - y,
        m_bufferPosition.z() - z
    };

    return { lightVector, paintingData.initialLightingData.colorVector, lightVector.length() };
}

bool PointLightSource::setX(float x)
{
    if (m_position.x() == x)
        return false;
    m_position.setX(x);
    return true;
}

bool PointLightSource::setY(float y)
{
    if (m_position.y() == y)
        return false;
    m_position.setY(y);
    return true;
}

bool PointLightSource::setZ(float z)
{
    if (m_position.z() == z)
        return false;
    m_position.setZ(z);
    return true;
}

TextStream& PointLightSource::externalRepresentation(TextStream& ts) const
{
    ts << "[type=POINT-LIGHT] ";
    ts << "[position=\"" << position() << "\"]";
    return ts;
}

}; // namespace WebCore
