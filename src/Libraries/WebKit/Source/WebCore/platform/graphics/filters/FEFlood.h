/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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

class FEFlood : public FilterEffect {
public:
    WEBCORE_EXPORT static Ref<FEFlood> create(const Color& floodColor, float floodOpacity, DestinationColorSpace = DestinationColorSpace::SRGB());

    bool operator==(const FEFlood&) const;

    const Color& floodColor() const { return m_floodColor; }
    bool setFloodColor(const Color&);

    float floodOpacity() const { return m_floodOpacity; }
    bool setFloodOpacity(float);

#if !USE(CG) && !USE(SKIA)
    // feFlood does not perform color interpolation of any kind, so the result is always in the current
    // color space regardless of the value of color-interpolation-filters.
    void setOperatingColorSpace(const DestinationColorSpace&) override { }
#endif

private:
    FEFlood(const Color& floodColor, float floodOpacity, DestinationColorSpace = DestinationColorSpace::SRGB());

    bool operator==(const FilterEffect& other) const override { return areEqual<FEFlood>(*this, other); }

    unsigned numberOfEffectInputs() const override { return 0; }

    FloatRect calculateImageRect(const Filter&, std::span<const FloatRect> inputImageRects, const FloatRect& primitiveSubregion) const override;

    std::unique_ptr<FilterEffectApplier> createSoftwareApplier() const override;

    WTF::TextStream& externalRepresentation(WTF::TextStream&, FilterRepresentation) const override;

    Color m_floodColor;
    float m_floodOpacity;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_FILTER_FUNCTION(FEFlood)
