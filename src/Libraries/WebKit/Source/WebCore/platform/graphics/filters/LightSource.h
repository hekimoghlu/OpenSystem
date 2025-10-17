/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 14, 2022.
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

#include "FloatPoint3D.h"
#include <wtf/RefCounted.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

enum class LightType : uint8_t {
    LS_DISTANT,
    LS_POINT,
    LS_SPOT
};

class Filter;
class FilterImage;

class LightSource : public RefCounted<LightSource> {
public:
    struct ComputedLightingData {
        FloatPoint3D lightVector;
        FloatPoint3D colorVector;
        float lightVectorLength;
    };

    struct PaintingData {
        ComputedLightingData initialLightingData;
        FloatPoint3D directionVector;
        float coneCutOffLimit;
        float coneFullLight;
    };

    LightSource(LightType type)
        : m_type(type)
    { }

    virtual ~LightSource() = default;

    virtual bool operator==(const LightSource& other) const
    {
        return m_type == other.m_type;
    }

    LightType type() const { return m_type; }
    virtual WTF::TextStream& externalRepresentation(WTF::TextStream&) const = 0;

    virtual void initPaintingData(const Filter&, const FilterImage& result, PaintingData&) const = 0;
    // z is a float number, since it is the alpha value scaled by a user
    // specified "surfaceScale" constant, which type is <number> in the SVG standard.
    // x and y are in the coordinates of the FilterEffect's buffer.
    virtual ComputedLightingData computePixelLightingData(const PaintingData&, int x, int y, float z) const = 0;

    virtual bool setAzimuth(float) { return false; }
    virtual bool setElevation(float) { return false; }
    
    // These are in user space coordinates.
    virtual bool setX(float) { return false; }
    virtual bool setY(float) { return false; }
    virtual bool setZ(float) { return false; }
    virtual bool setPointsAtX(float) { return false; }
    virtual bool setPointsAtY(float) { return false; }
    virtual bool setPointsAtZ(float) { return false; }
    
    virtual bool setSpecularExponent(float) { return false; }
    virtual bool setLimitingConeAngle(float) { return false; }

protected:
    template<typename LightSourceType>
    static bool areEqual(const LightSourceType& a, const LightSource& b)
    {
        auto* bType = dynamicDowncast<LightSourceType>(b);
        return bType && a.operator==(*bType);
    }

private:
    LightType m_type;
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_LIGHTSOURCE(ClassName, Type) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ClassName) \
    static bool isType(const WebCore::LightSource& source) { return source.type() == WebCore::Type; } \
SPECIALIZE_TYPE_TRAITS_END()
