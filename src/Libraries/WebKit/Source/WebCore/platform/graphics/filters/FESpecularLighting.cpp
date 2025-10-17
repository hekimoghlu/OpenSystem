/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 28, 2024.
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
#include "FESpecularLighting.h"

#include "ImageBuffer.h"
#include "LightSource.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<FESpecularLighting> FESpecularLighting::create(const Color& lightingColor, float surfaceScale, float specularConstant, float specularExponent, float kernelUnitLengthX, float kernelUnitLengthY, Ref<LightSource>&& lightSource, DestinationColorSpace colorSpace)
{
    return adoptRef(*new FESpecularLighting(lightingColor, surfaceScale, specularConstant, specularExponent, kernelUnitLengthX, kernelUnitLengthY, WTFMove(lightSource), colorSpace));
}

FESpecularLighting::FESpecularLighting(const Color& lightingColor, float surfaceScale, float specularConstant, float specularExponent, float kernelUnitLengthX, float kernelUnitLengthY, Ref<LightSource>&& lightSource, DestinationColorSpace colorSpace)
    : FELighting(FilterEffect::Type::FESpecularLighting, lightingColor, surfaceScale, 0, specularConstant, specularExponent, kernelUnitLengthX, kernelUnitLengthY, WTFMove(lightSource), colorSpace)
{
}

bool FESpecularLighting::setSpecularConstant(float specularConstant)
{
    specularConstant = std::max(specularConstant, 0.0f);
    if (m_specularConstant == specularConstant)
        return false;
    m_specularConstant = specularConstant;
    return true;
}

bool FESpecularLighting::setSpecularExponent(float specularExponent)
{
    specularExponent = clampTo<float>(specularExponent, 1.0f, 128.0f);
    if (m_specularExponent == specularExponent)
        return false;
    m_specularExponent = specularExponent;
    return true;
}

TextStream& FESpecularLighting::externalRepresentation(TextStream& ts, FilterRepresentation representation) const
{
    ts << indent << "[feSpecularLighting";
    FilterEffect::externalRepresentation(ts, representation);
    
    ts << " surfaceScale=\"" << m_surfaceScale << "\"";
    ts << " specualConstant=\"" << m_specularConstant << "\"";
    ts << " specularExponent=\"" << m_specularExponent << "\"";

    ts << "]\n";
    return ts;
}

} // namespace WebCore
