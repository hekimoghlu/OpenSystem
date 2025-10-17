/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
#include "DistantLightSource.h"

#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<DistantLightSource> DistantLightSource::create(float azimuth, float elevation)
{
    return adoptRef(*new DistantLightSource(azimuth, elevation));
}

DistantLightSource::DistantLightSource(float azimuth, float elevation)
    : LightSource(LightType::LS_DISTANT)
    , m_azimuth(azimuth)
    , m_elevation(elevation)
{
}

bool DistantLightSource::operator==(const DistantLightSource& other) const
{
    return LightSource::operator==(other)
        && m_azimuth == other.m_azimuth
        && m_elevation == other.m_elevation;
}

void DistantLightSource::initPaintingData(const Filter&, const FilterImage&, PaintingData& paintingData) const
{
    float azimuth = deg2rad(m_azimuth);
    float elevation = deg2rad(m_elevation);
    paintingData.initialLightingData.lightVector = {
        std::cos(azimuth) * std::cos(elevation),
        std::sin(azimuth) * std::cos(elevation),
        std::sin(elevation)
    };
    paintingData.initialLightingData.lightVectorLength = 1;
}

LightSource::ComputedLightingData DistantLightSource::computePixelLightingData(const PaintingData& paintingData, int, int, float) const
{
    return paintingData.initialLightingData;
}

bool DistantLightSource::setAzimuth(float azimuth)
{
    if (m_azimuth == azimuth)
        return false;
    m_azimuth = azimuth;
    return true;
}

bool DistantLightSource::setElevation(float elevation)
{
    if (m_elevation == elevation)
        return false;
    m_elevation = elevation;
    return true;
}

TextStream& DistantLightSource::externalRepresentation(TextStream& ts) const
{
    ts << "[type=DISTANT-LIGHT] ";
    ts << "[azimuth=\"" << azimuth() << "\"]";
    ts << "[elevation=\"" << elevation() << "\"]";
    return ts;
}

} // namespace WebCore
