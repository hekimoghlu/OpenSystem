/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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
#include "FEDiffuseLighting.h"

#include "ImageBuffer.h"
#include "LightSource.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<FEDiffuseLighting> FEDiffuseLighting::create(const Color& lightingColor, float surfaceScale, float diffuseConstant, float kernelUnitLengthX, float kernelUnitLengthY, Ref<LightSource>&& lightSource, DestinationColorSpace colorSpace)
{
    return adoptRef(*new FEDiffuseLighting(lightingColor, surfaceScale, diffuseConstant, kernelUnitLengthX, kernelUnitLengthY, WTFMove(lightSource), colorSpace));
}

FEDiffuseLighting::FEDiffuseLighting(const Color& lightingColor, float surfaceScale, float diffuseConstant, float kernelUnitLengthX, float kernelUnitLengthY, Ref<LightSource>&& lightSource, DestinationColorSpace colorSpace)
    : FELighting(FilterEffect::Type::FEDiffuseLighting, lightingColor, surfaceScale, diffuseConstant, 0, 0, kernelUnitLengthX, kernelUnitLengthY, WTFMove(lightSource), colorSpace)
{
}

bool FEDiffuseLighting::setDiffuseConstant(float diffuseConstant)
{
    diffuseConstant = std::max(diffuseConstant, 0.0f);
    if (m_diffuseConstant == diffuseConstant)
        return false;
    m_diffuseConstant = diffuseConstant;
    return true;
}

TextStream& FEDiffuseLighting::externalRepresentation(TextStream& ts, FilterRepresentation representation) const
{
    ts << indent << "[feDiffuseLighting";
    FilterEffect::externalRepresentation(ts, representation);

    ts << " surfaceScale=\"" << m_surfaceScale << "\"";
    ts << " diffuseConstant=\"" << m_diffuseConstant << "\"";
    ts << " kernelUnitLength=\"" << m_kernelUnitLengthX << ", " << m_kernelUnitLengthY << "\"";

    ts << "]\n";
    return ts;
}

} // namespace WebCore
