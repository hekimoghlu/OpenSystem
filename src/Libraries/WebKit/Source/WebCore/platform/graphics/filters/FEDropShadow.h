/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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
#include "FilterEffect.h"

namespace WebCore {
    
class FEDropShadow : public FilterEffect {
public:
    WEBCORE_EXPORT static Ref<FEDropShadow> create(float stdX, float stdY, float dx, float dy, const Color& shadowColor, float shadowOpacity, DestinationColorSpace = DestinationColorSpace::SRGB());

    bool operator==(const FEDropShadow&) const;

    float stdDeviationX() const { return m_stdX; }
    bool setStdDeviationX(float);

    float stdDeviationY() const { return m_stdY; }
    bool setStdDeviationY(float);

    float dx() const { return m_dx; }
    bool setDx(float);

    float dy() const { return m_dy; }
    bool setDy(float);

    const Color& shadowColor() const { return m_shadowColor; } 
    bool setShadowColor(const Color&);

    float shadowOpacity() const { return m_shadowOpacity; }
    bool setShadowOpacity(float);

    static IntOutsets calculateOutsets(const FloatSize& offset, const FloatSize& stdDeviation);

#if USE(CAIRO)
    void setOperatingColorSpace(const DestinationColorSpace&) override { }
#endif

private:
    FEDropShadow(float stdX, float stdY, float dx, float dy, const Color& shadowColor, float shadowOpacity, DestinationColorSpace);

    bool operator==(const FilterEffect& other) const override { return areEqual<FEDropShadow>(*this, other); }

    FloatRect calculateImageRect(const Filter&, std::span<const FloatRect> inputImageRects, const FloatRect& primitiveSubregion) const override;

    OptionSet<FilterRenderingMode> supportedFilterRenderingModes() const override;

    std::unique_ptr<FilterEffectApplier> createAcceleratedApplier() const override;
    std::unique_ptr<FilterEffectApplier> createSoftwareApplier() const override;
    std::optional<GraphicsStyle> createGraphicsStyle(GraphicsContext&, const Filter&) const override;

    WTF::TextStream& externalRepresentation(WTF::TextStream&, FilterRepresentation) const override;

    float m_stdX;
    float m_stdY;
    float m_dx;
    float m_dy;
    Color m_shadowColor;
    float m_shadowOpacity;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_FILTER_FUNCTION(FEDropShadow)
