/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 27, 2022.
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
#include "FELightingSoftwareApplier.h"

#include "FELighting.h"
#include "FELightingSoftwareApplierInlines.h"
#include "Filter.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FELightingSoftwareApplier);

void FELightingSoftwareApplier::setPixelInternal(int offset, const LightingData& data, const LightSource::PaintingData& paintingData, int x, int y, float factorX, float factorY, IntSize normal2DVector, float alpha)
{
    float z = alpha * data.surfaceScale;
    LightSource::ComputedLightingData lightingData = data.lightSource->computePixelLightingData(paintingData, x, y, z);

    float lightStrength;
    if (normal2DVector.isZero()) {
        // Normal vector is (0, 0, 1). This is a quite frequent case.
        if (data.filterType == FilterEffect::Type::FEDiffuseLighting)
            lightStrength = data.diffuseConstant * lightingData.lightVector.z() / lightingData.lightVectorLength;
        else {
            FloatPoint3D halfwayVector = {
                lightingData.lightVector.x(),
                lightingData.lightVector.y(),
                lightingData.lightVector.z() + lightingData.lightVectorLength
            };
            float halfwayVectorLength = halfwayVector.length();
            if (data.specularExponent == 1)
                lightStrength = data.specularConstant * halfwayVector.z() / halfwayVectorLength;
            else
                lightStrength = data.specularConstant * powf(halfwayVector.z() / halfwayVectorLength, data.specularExponent);
        }
    } else {
        FloatPoint3D normalVector = {
            factorX * normal2DVector.width() * data.surfaceScale,
            factorY * normal2DVector.height() * data.surfaceScale,
            1.0f
        };
        float normalVectorLength = normalVector.length();

        if (data.filterType == FilterEffect::Type::FEDiffuseLighting)
            lightStrength = data.diffuseConstant * (normalVector * lightingData.lightVector) / (normalVectorLength * lightingData.lightVectorLength);
        else {
            FloatPoint3D halfwayVector = {
                lightingData.lightVector.x(),
                lightingData.lightVector.y(),
                lightingData.lightVector.z() + lightingData.lightVectorLength
            };
            float halfwayVectorLength = halfwayVector.length();
            if (data.specularExponent == 1)
                lightStrength = data.specularConstant * (normalVector * halfwayVector) / (normalVectorLength * halfwayVectorLength);
            else
                lightStrength = data.specularConstant * powf((normalVector * halfwayVector) / (normalVectorLength * halfwayVectorLength), data.specularExponent);
        }
    }

    if (lightStrength > 1)
        lightStrength = 1;
    if (lightStrength < 0)
        lightStrength = 0;

    uint8_t pixelValue[3] = {
        static_cast<uint8_t>(lightStrength * lightingData.colorVector.x() * 255.0f),
        static_cast<uint8_t>(lightStrength * lightingData.colorVector.y() * 255.0f),
        static_cast<uint8_t>(lightStrength * lightingData.colorVector.z() * 255.0f)
    };
    
    data.pixels->setRange({ pixelValue }, offset);
}

void FELightingSoftwareApplier::setPixel(int offset, const LightingData& data, const LightSource::PaintingData& paintingData, int x, int y, float factorX, float factorY, IntSize normal2DVector)
{
    setPixelInternal(offset, data, paintingData, x, y, factorX, factorY, normal2DVector, data.pixels->item(offset + cAlphaChannelOffset));
}

void FELightingSoftwareApplier::applyPlatform(const LightingData& data) const
{
    LightSource::PaintingData paintingData;

    auto [r, g, b, a] = data.lightingColor.toResolvedColorComponentsInColorSpace(*data.operatingColorSpace);
    paintingData.initialLightingData.colorVector = FloatPoint3D(r, g, b);

    data.lightSource->initPaintingData(Ref { *data.filter }, Ref { *data.result }, paintingData);

    // Top left.
    int offset = 0;
    setPixel(offset, data, paintingData, 0, 0, cFactor2div3, cFactor2div3, data.topLeftNormal(offset));

    // Top right.
    offset = data.widthMultipliedByPixelSize - cPixelSize;
    setPixel(offset, data, paintingData, data.width - 1, 0, cFactor2div3, cFactor2div3, data.topRightNormal(offset));

    // Bottom left.
    offset = (data.height - 1) * data.widthMultipliedByPixelSize;
    setPixel(offset, data, paintingData, 0, data.height - 1, cFactor2div3, cFactor2div3, data.bottomLeftNormal(offset));

    // Bottom right.
    offset = data.height * data.widthMultipliedByPixelSize - cPixelSize;
    setPixel(offset, data, paintingData, data.width - 1, data.height - 1, cFactor2div3, cFactor2div3, data.bottomRightNormal(offset));

    if (data.width >= 3) {
        // Top row.
        offset = cPixelSize;
        for (int x = 1; x < data.width - 1; ++x, offset += cPixelSize)
            setPixel(offset, data, paintingData, x, 0, cFactor1div3, cFactor1div2, data.topRowNormal(offset));

        // Bottom row.
        offset = (data.height - 1) * data.widthMultipliedByPixelSize + cPixelSize;
        for (int x = 1; x < data.width - 1; ++x, offset += cPixelSize)
            setPixel(offset, data, paintingData, x, data.height - 1, cFactor1div3, cFactor1div2, data.bottomRowNormal(offset));
    }

    if (data.height >= 3) {
        // Left column.
        offset = data.widthMultipliedByPixelSize;
        for (int y = 1; y < data.height - 1; ++y, offset += data.widthMultipliedByPixelSize)
            setPixel(offset, data, paintingData, 0, y, cFactor1div2, cFactor1div3, data.leftColumnNormal(offset));

        // Right column.
        offset = 2 * data.widthMultipliedByPixelSize - cPixelSize;
        for (int y = 1; y < data.height - 1; ++y, offset += data.widthMultipliedByPixelSize)
            setPixel(offset, data, paintingData, data.width - 1, y, cFactor1div2, cFactor1div3, data.rightColumnNormal(offset));
    }

    if (data.width >= 3 && data.height >= 3) {
        // Interior pixels.
        applyPlatformParallel(data, paintingData);
    }

    int lastPixel = data.widthMultipliedByPixelSize * data.height;
    if (data.filterType == FilterEffect::Type::FEDiffuseLighting) {
        for (int i = cAlphaChannelOffset; i < lastPixel; i += cPixelSize)
            data.pixels->set(i, cOpaqueAlpha);
    } else {
        for (int i = 0; i < lastPixel; i += cPixelSize) {
            uint8_t a1 = data.pixels->item(i);
            uint8_t a2 = data.pixels->item(i + 1);
            uint8_t a3 = data.pixels->item(i + 2);
            // alpha set to set to max(a1, a2, a3)
            data.pixels->set(i + 3, a1 >= a2 ? (a1 >= a3 ? a1 : a3) : (a2 >= a3 ? a2 : a3));
        }
    }
}

bool FELightingSoftwareApplier::apply(const Filter& filter, const FilterImageVector& inputs, FilterImage& result) const
{
    auto& input = inputs[0].get();

    auto destinationPixelBuffer = result.pixelBuffer(AlphaPremultiplication::Premultiplied);
    if (!destinationPixelBuffer)
        return false;

    auto effectDrawingRect = result.absoluteImageRectRelativeTo(input);
    input.copyPixelBuffer(*destinationPixelBuffer, effectDrawingRect);

    // FIXME: support kernelUnitLengths other than (1,1). The issue here is that the W3
    // standard has no test case for them, and other browsers (like Firefox) has strange
    // output for various kernelUnitLengths, and I am not sure they are reliable.
    // Anyway, feConvolveMatrix should also use the implementation

    auto size = IntSize(result.absoluteImageRect().size());

    // FIXME: do something if width or height (or both) is 1 pixel.
    // The W3 spec does not define this case. Now the filter just returns.
    if (size.width() <= 2 || size.height() <= 2)
        return true;

    LightingData data;
    data.filter = &filter;
    data.result = &result;
    data.filterType = m_effect.filterType();
    data.lightingColor = m_effect.lightingColor();
    data.surfaceScale = m_effect.surfaceScale() / 255.0f;
    data.diffuseConstant = m_effect.diffuseConstant();
    data.specularConstant = m_effect.specularConstant();
    data.specularExponent = m_effect.specularExponent();
    data.lightSource = m_effect.lightSource().ptr();
    data.operatingColorSpace = &m_effect.operatingColorSpace();

    data.pixels = destinationPixelBuffer;
    data.widthMultipliedByPixelSize = size.width() * cPixelSize;
    data.width = size.width();
    data.height = size.height();

    applyPlatform(data);
    return true;
}

} // namespace WebCore
