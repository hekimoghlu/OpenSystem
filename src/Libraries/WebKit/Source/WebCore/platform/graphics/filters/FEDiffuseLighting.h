/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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

#include "FELighting.h"

namespace WebCore {

class LightSource;

class FEDiffuseLighting : public FELighting {
public:
    WEBCORE_EXPORT static Ref<FEDiffuseLighting> create(const Color& lightingColor, float surfaceScale, float diffuseConstant, float kernelUnitLengthX, float kernelUnitLengthY, Ref<LightSource>&&, DestinationColorSpace = DestinationColorSpace::SRGB());

    bool operator==(const FEDiffuseLighting& other) const { return FELighting::operator==(other); }

    float diffuseConstant() const { return m_diffuseConstant; }
    bool setDiffuseConstant(float);

    WTF::TextStream& externalRepresentation(WTF::TextStream&, FilterRepresentation) const override;

private:
    FEDiffuseLighting(const Color& lightingColor, float surfaceScale, float diffuseConstant, float kernelUnitLengthX, float kernelUnitLengthY, Ref<LightSource>&&, DestinationColorSpace);

    bool operator==(const FilterEffect& other) const override { return areEqual<FEDiffuseLighting>(*this, other); }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_FILTER_FUNCTION(FEDiffuseLighting)
