/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 22, 2023.
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
#include "SpotLightSource.h"

#include "Filter.h"
#include "FilterImage.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

// spot-light edge darkening depends on an absolute treshold
// according to the SVG 1.1 SE light regression tests
static const float antialiasThreshold = 0.016f;

Ref<SpotLightSource> SpotLightSource::create(const FloatPoint3D& position, const FloatPoint3D& direction, float specularExponent, float limitingConeAngle)
{
    return adoptRef(*new SpotLightSource(position, direction, specularExponent, limitingConeAngle));
}

SpotLightSource::SpotLightSource(const FloatPoint3D& position, const FloatPoint3D& pointsAt, float specularExponent, float limitingConeAngle)
    : LightSource(LightType::LS_SPOT)
    , m_position(position)
    , m_pointsAt(pointsAt)
    , m_specularExponent(clampTo<float>(specularExponent, 1.0f, 128.0f))
    , m_limitingConeAngle(limitingConeAngle)
{
}

bool SpotLightSource::operator==(const SpotLightSource& other) const
{
    return LightSource::operator==(other)
        && m_position == other.m_position
        && m_pointsAt == other.m_pointsAt
        && m_specularExponent == other.m_specularExponent
        && m_limitingConeAngle == other.m_limitingConeAngle;
}

void SpotLightSource::initPaintingData(const Filter& filter, const FilterImage& result, PaintingData& paintingData) const
{
    auto position = filter.resolvedPoint3D(m_position);
    auto pointsAt = filter.resolvedPoint3D(m_pointsAt);

    auto absolutePosition = filter.scaledByFilterScale(position.xy());
    m_bufferPosition.setXY(result.mappedAbsolutePoint(absolutePosition));

    // To scale Z, map a point offset from position in the x direction by z.
    auto absoluteMappedZ = filter.scaledByFilterScale(FloatPoint { position.x() + position.z(), position.y() });
    m_bufferPosition.setZ(result.mappedAbsolutePoint(absoluteMappedZ).x() - m_bufferPosition.x());
    
    paintingData.directionVector = pointsAt - position;
    paintingData.directionVector.normalize();

    if (!m_limitingConeAngle) {
        paintingData.coneCutOffLimit = 0.0f;
        paintingData.coneFullLight = -antialiasThreshold;
    } else {
        float limitingConeAngle = m_limitingConeAngle;
        if (limitingConeAngle < 0.0f)
            limitingConeAngle = -limitingConeAngle;
        if (limitingConeAngle > 90.0f)
            limitingConeAngle = 90.0f;
        paintingData.coneCutOffLimit = cosf(deg2rad(180.0f - limitingConeAngle));
        paintingData.coneFullLight = paintingData.coneCutOffLimit - antialiasThreshold;
    }
}

LightSource::ComputedLightingData SpotLightSource::computePixelLightingData(const PaintingData& paintingData, int x, int y, float z) const
{
    FloatPoint3D lightVector = {
        m_bufferPosition.x() - x,
        m_bufferPosition.y() - y,
        m_bufferPosition.z() - z
    };
    float lightVectorLength = lightVector.length();

    float cosineOfAngle = (lightVector * paintingData.directionVector) / lightVectorLength;
    if (cosineOfAngle > paintingData.coneCutOffLimit) {
        // No light is produced, scanlines are not updated
        return { lightVector, { }, lightVectorLength };
    }

    // Set the color of the pixel
    float lightStrength;
    if (1.0f == m_specularExponent)
        lightStrength = -cosineOfAngle; // -cosineOfAngle ^ 1 == -cosineOfAngle
    else
        lightStrength = powf(-cosineOfAngle, m_specularExponent);

    if (cosineOfAngle > paintingData.coneFullLight)
        lightStrength *= (paintingData.coneCutOffLimit - cosineOfAngle) / (paintingData.coneCutOffLimit - paintingData.coneFullLight);

    if (lightStrength > 1.0f)
        lightStrength = 1.0f;

    return {
        lightVector,
        paintingData.initialLightingData.colorVector * lightStrength,
        lightVectorLength
    };
}

bool SpotLightSource::setX(float x)
{
    if (m_position.x() == x)
        return false;
    m_position.setX(x);
    return true;
}

bool SpotLightSource::setY(float y)
{
    if (m_position.y() == y)
        return false;
    m_position.setY(y);
    return true;
}

bool SpotLightSource::setZ(float z)
{
    if (m_position.z() == z)
        return false;
    m_position.setZ(z);
    return true;
}

bool SpotLightSource::setPointsAtX(float pointsAtX)
{
    if (m_pointsAt.x() == pointsAtX)
        return false;
    m_pointsAt.setX(pointsAtX);
    return true;
}

bool SpotLightSource::setPointsAtY(float pointsAtY)
{
    if (m_pointsAt.y() == pointsAtY)
        return false;
    m_pointsAt.setY(pointsAtY);
    return true;
}

bool SpotLightSource::setPointsAtZ(float pointsAtZ)
{
    if (m_pointsAt.z() == pointsAtZ)
        return false;
    m_pointsAt.setZ(pointsAtZ);
    return true;
}

bool SpotLightSource::setSpecularExponent(float specularExponent)
{
    specularExponent = clampTo<float>(specularExponent, 1.0f, 128.0f);
    if (m_specularExponent == specularExponent)
        return false;
    m_specularExponent = specularExponent;
    return true;
}

bool SpotLightSource::setLimitingConeAngle(float limitingConeAngle)
{
    if (m_limitingConeAngle == limitingConeAngle)
        return false;
    m_limitingConeAngle = limitingConeAngle;
    return true;
}

TextStream& SpotLightSource::externalRepresentation(TextStream& ts) const
{
    ts << "[type=SPOT-LIGHT] ";
    ts << "[position=\"" << position() << "\"]";
    ts << "[direction=\"" << direction() << "\"]";
    ts << "[specularExponent=\"" << specularExponent() << "\"]";
    ts << "[limitingConeAngle=\"" << limitingConeAngle() << "\"]";
    return ts;
}

}; // namespace WebCore
