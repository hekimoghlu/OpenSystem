/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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

#include "Color.h"
#include "DistantLightSource.h"
#include "FilterEffect.h"
#include "PointLightSource.h"
#include "SpotLightSource.h"

namespace WebCore {

struct FELightingPaintingDataForNeon;

class FELighting : public FilterEffect {
public:
    bool operator==(const FELighting&) const;

    const Color& lightingColor() const { return m_lightingColor; }
    bool setLightingColor(const Color&);

    float surfaceScale() const { return m_surfaceScale; }
    bool setSurfaceScale(float);

    float diffuseConstant() const { return m_diffuseConstant; }
    float specularConstant() const { return m_specularConstant; }
    float specularExponent() const { return m_specularExponent; }

    float kernelUnitLengthX() const { return m_kernelUnitLengthX; }
    bool setKernelUnitLengthX(float);

    float kernelUnitLengthY() const { return m_kernelUnitLengthY; }
    bool setKernelUnitLengthY(float);

    Ref<LightSource> lightSource() const { return m_lightSource; }

protected:
    FELighting(Type, const Color& lightingColor, float surfaceScale, float diffuseConstant, float specularConstant, float specularExponent, float kernelUnitLengthX, float kernelUnitLengthY, Ref<LightSource>&&, DestinationColorSpace);

    using FilterEffect::operator==;

    FloatRect calculateImageRect(const Filter&, std::span<const FloatRect> inputImageRects, const FloatRect& primitiveSubregion) const override;

    std::unique_ptr<FilterEffectApplier> createSoftwareApplier() const override;

    Color m_lightingColor;
    float m_surfaceScale;
    float m_diffuseConstant;
    float m_specularConstant;
    float m_specularExponent;
    float m_kernelUnitLengthX;
    float m_kernelUnitLengthY;
    Ref<LightSource> m_lightSource;
};

} // namespace WebCore
