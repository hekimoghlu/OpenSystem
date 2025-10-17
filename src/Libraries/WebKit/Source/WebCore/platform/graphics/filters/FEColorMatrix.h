/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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

#include "FilterEffect.h"
#include <wtf/Vector.h>

namespace WebCore {

enum class ColorMatrixType : uint8_t {
    FECOLORMATRIX_TYPE_UNKNOWN          = 0,
    FECOLORMATRIX_TYPE_MATRIX           = 1,
    FECOLORMATRIX_TYPE_SATURATE         = 2,
    FECOLORMATRIX_TYPE_HUEROTATE        = 3,
    FECOLORMATRIX_TYPE_LUMINANCETOALPHA = 4
};

class FEColorMatrix : public FilterEffect {
public:
    WEBCORE_EXPORT static Ref<FEColorMatrix> create(ColorMatrixType, Vector<float>&&, DestinationColorSpace = DestinationColorSpace::SRGB());

    bool operator==(const FEColorMatrix&) const;

    ColorMatrixType type() const { return m_type; }
    bool setType(ColorMatrixType);

    const Vector<float>& values() const { return m_values; }
    bool setValues(const Vector<float>&);

    static void calculateSaturateComponents(std::span<float, 9> components, float value);
    static void calculateHueRotateComponents(std::span<float, 9> components, float value);
    static Vector<float> normalizedFloats(const Vector<float>& values);

private:
    FEColorMatrix(ColorMatrixType, Vector<float>&&, DestinationColorSpace);

    bool operator==(const FilterEffect& other) const override { return areEqual<FEColorMatrix>(*this, other); }

    bool resultIsAlphaImage(const FilterImageVector& inputs) const override;

    OptionSet<FilterRenderingMode> supportedFilterRenderingModes() const override;

    std::unique_ptr<FilterEffectApplier> createAcceleratedApplier() const override;
    std::unique_ptr<FilterEffectApplier> createSoftwareApplier() const override;
    std::optional<GraphicsStyle> createGraphicsStyle(GraphicsContext&, const Filter&) const override;

    WTF::TextStream& externalRepresentation(WTF::TextStream&, FilterRepresentation) const override;

    ColorMatrixType m_type;
    Vector<float> m_values;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_FILTER_FUNCTION(FEColorMatrix)
