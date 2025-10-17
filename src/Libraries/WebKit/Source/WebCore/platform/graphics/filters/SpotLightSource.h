/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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

#include "LightSource.h"
#include <wtf/Ref.h>

namespace WebCore {

class SpotLightSource : public LightSource {
public:
    WEBCORE_EXPORT static Ref<SpotLightSource> create(const FloatPoint3D& position, const FloatPoint3D& pointsAt, float specularExponent, float limitingConeAngle);

    bool operator==(const SpotLightSource&) const;

    const FloatPoint3D& position() const { return m_position; }
    const FloatPoint3D& direction() const { return m_pointsAt; }
    float specularExponent() const { return m_specularExponent; }
    float limitingConeAngle() const { return m_limitingConeAngle; }

    bool setX(float) override;
    bool setY(float) override;
    bool setZ(float) override;
    bool setPointsAtX(float) override;
    bool setPointsAtY(float) override;
    bool setPointsAtZ(float) override;

    bool setSpecularExponent(float) override;
    bool setLimitingConeAngle(float) override;

    void initPaintingData(const Filter&, const FilterImage& result, PaintingData&) const override;
    ComputedLightingData computePixelLightingData(const PaintingData&, int x, int y, float z) const final;

    WTF::TextStream& externalRepresentation(WTF::TextStream&) const override;

private:
    SpotLightSource(const FloatPoint3D& position, const FloatPoint3D& direction, float specularExponent, float limitingConeAngle);

    bool operator==(const LightSource& other) const override { return areEqual<SpotLightSource>(*this, other); }

    FloatPoint3D m_position;
    FloatPoint3D m_pointsAt;

    mutable FloatPoint3D m_bufferPosition;

    float m_specularExponent;
    float m_limitingConeAngle;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_LIGHTSOURCE(SpotLightSource, LightType::LS_SPOT)
