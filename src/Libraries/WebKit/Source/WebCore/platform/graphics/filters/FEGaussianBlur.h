/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 16, 2024.
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

#include "FEConvolveMatrix.h"
#include "FilterEffect.h"

namespace WebCore {

class FEGaussianBlur : public FilterEffect {
public:
    WEBCORE_EXPORT static Ref<FEGaussianBlur> create(float x, float y, EdgeModeType, DestinationColorSpace = DestinationColorSpace::SRGB());

    bool operator==(const FEGaussianBlur&) const;

    float stdDeviationX() const { return m_stdX; }
    bool setStdDeviationX(float);

    float stdDeviationY() const { return m_stdY; }
    bool setStdDeviationY(float);

    EdgeModeType edgeMode() const { return m_edgeMode; }
    bool setEdgeMode(EdgeModeType);

    static IntSize calculateKernelSize(const Filter&, FloatSize stdDeviation);
    static IntSize calculateUnscaledKernelSize(FloatSize stdDeviation);
    static IntSize calculateOutsetSize(FloatSize stdDeviation);

    static IntOutsets calculateOutsets(const FloatSize& stdDeviation);

private:
    FEGaussianBlur(float x, float y, EdgeModeType, DestinationColorSpace);

    bool operator==(const FilterEffect& other) const override { return areEqual<FEGaussianBlur>(*this, other); }

    FloatRect calculateImageRect(const Filter&, std::span<const FloatRect> inputImageRects, const FloatRect& primitiveSubregion) const override;

    bool resultIsAlphaImage(const FilterImageVector& inputs) const override;

    OptionSet<FilterRenderingMode> supportedFilterRenderingModes() const override;

    std::unique_ptr<FilterEffectApplier> createAcceleratedApplier() const override;
    std::unique_ptr<FilterEffectApplier> createSoftwareApplier() const override;
    std::optional<GraphicsStyle> createGraphicsStyle(GraphicsContext&, const Filter&) const override;

    WTF::TextStream& externalRepresentation(WTF::TextStream&, FilterRepresentation) const override;

    float m_stdX;
    float m_stdY;
    EdgeModeType m_edgeMode;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_FILTER_FUNCTION(FEGaussianBlur)
