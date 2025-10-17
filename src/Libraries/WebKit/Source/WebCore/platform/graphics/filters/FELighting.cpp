/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 22, 2022.
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
#include "FELighting.h"

#include "FELightingNeonParallelApplier.h"
#include "FELightingSoftwareParallelApplier.h"
#include "Filter.h"

namespace WebCore {

FELighting::FELighting(Type type, const Color& lightingColor, float surfaceScale, float diffuseConstant, float specularConstant, float specularExponent, float kernelUnitLengthX, float kernelUnitLengthY, Ref<LightSource>&& lightSource, DestinationColorSpace colorSpace)
    : FilterEffect(type, colorSpace)
    , m_lightingColor(lightingColor)
    , m_surfaceScale(surfaceScale)
    , m_diffuseConstant(std::max(diffuseConstant, 0.0f))
    , m_specularConstant(std::max(specularConstant, 0.0f))
    , m_specularExponent(clampTo<float>(specularExponent, 1.0f, 128.0f))
    , m_kernelUnitLengthX(kernelUnitLengthX)
    , m_kernelUnitLengthY(kernelUnitLengthY)
    , m_lightSource(WTFMove(lightSource))
{
}

bool FELighting::operator==(const FELighting& other) const
{
    return FilterEffect::operator==(other)
        && m_lightingColor == other.m_lightingColor
        && m_surfaceScale == other.m_surfaceScale
        && m_diffuseConstant == other.m_diffuseConstant
        && m_specularConstant == other.m_specularConstant
        && m_specularExponent == other.m_specularExponent
        && m_kernelUnitLengthX == other.m_kernelUnitLengthX
        && m_kernelUnitLengthY == other.m_kernelUnitLengthY
        && m_lightSource.get() == other.m_lightSource.get();
}

bool FELighting::setSurfaceScale(float surfaceScale)
{
    if (m_surfaceScale == surfaceScale)
        return false;

    m_surfaceScale = surfaceScale;
    return true;
}

bool FELighting::setLightingColor(const Color& lightingColor)
{
    if (m_lightingColor == lightingColor)
        return false;

    m_lightingColor = lightingColor;
    return true;
}

bool FELighting::setKernelUnitLengthX(float kernelUnitLengthX)
{
    if (m_kernelUnitLengthX == kernelUnitLengthX)
        return false;

    m_kernelUnitLengthX = kernelUnitLengthX;
    return true;
}

bool FELighting::setKernelUnitLengthY(float kernelUnitLengthY)
{
    if (m_kernelUnitLengthY == kernelUnitLengthY)
        return false;

    m_kernelUnitLengthY = kernelUnitLengthY;
    return true;
}

FloatRect FELighting::calculateImageRect(const Filter& filter, std::span<const FloatRect>, const FloatRect& primitiveSubregion) const
{
    return filter.maxEffectRect(primitiveSubregion);
}

std::unique_ptr<FilterEffectApplier> FELighting::createSoftwareApplier() const
{
#if (CPU(ARM_NEON) && CPU(ARM_TRADITIONAL) && COMPILER(GCC_COMPATIBLE))
    return FilterEffectApplier::create<FELightingNeonParallelApplier>(*this);
#else
    return FilterEffectApplier::create<FELightingSoftwareParallelApplier>(*this);
#endif
}

} // namespace WebCore
