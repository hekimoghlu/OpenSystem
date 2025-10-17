/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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

#include "DestinationColorSpace.h"
#include "FilterEffectApplier.h"
#include "FilterFunction.h"
#include "FilterImageVector.h"

namespace WTF {
class TextStream;
}

namespace WebCore {

class Filter;
class FilterEffectGeometry;
class FilterResults;

class FilterEffect : public FilterFunction {
    using FilterFunction::apply;

public:
    virtual bool operator==(const FilterEffect&) const;

    const DestinationColorSpace& operatingColorSpace() const { return m_operatingColorSpace; }
    virtual void setOperatingColorSpace(const DestinationColorSpace& colorSpace) { m_operatingColorSpace = colorSpace; }

    unsigned numberOfImageInputs() const { return filterType() == FilterEffect::Type::SourceGraphic ? 1 : numberOfEffectInputs(); }
    FilterImageVector takeImageInputs(FilterImageVector& stack) const;

    RefPtr<FilterImage> apply(const Filter&, const FilterImageVector& inputs, FilterResults&, const std::optional<FilterEffectGeometry>& = std::nullopt);
    FilterStyle createFilterStyle(GraphicsContext&, const Filter&, const FilterStyle& input, const std::optional<FilterEffectGeometry>& = std::nullopt) const;

    WTF::TextStream& externalRepresentation(WTF::TextStream&, FilterRepresentation) const override;

protected:
    explicit FilterEffect(Type, DestinationColorSpace = DestinationColorSpace::SRGB(), std::optional<RenderingResourceIdentifier> = std::nullopt);

    template<typename FilterEffectType>
    static bool areEqual(const FilterEffectType& a, const FilterEffect& b)
    {
        auto* bType = dynamicDowncast<FilterEffectType>(b);
        return bType && a.operator==(*bType);
    }

    virtual unsigned numberOfEffectInputs() const { return 1; }

    FloatRect calculatePrimitiveSubregion(const Filter&, std::span<const FloatRect> inputPrimitiveSubregions, const std::optional<FilterEffectGeometry>&) const;

    virtual FloatRect calculateImageRect(const Filter&, std::span<const FloatRect> inputImageRects, const FloatRect& primitiveSubregion) const;

    // Solid black image with different alpha values.
    virtual bool resultIsAlphaImage(const FilterImageVector&) const { return false; }

    virtual bool resultIsValidPremultiplied() const { return true; }

    virtual const DestinationColorSpace& resultColorSpace(const FilterImageVector&) const { return m_operatingColorSpace; }

    virtual void transformInputsColorSpace(const FilterImageVector& inputs) const;
    
    void correctPremultipliedInputs(const FilterImageVector& inputs) const;

    std::unique_ptr<FilterEffectApplier> createApplier(const Filter&) const;

    virtual std::unique_ptr<FilterEffectApplier> createAcceleratedApplier() const { return nullptr; }
    virtual std::unique_ptr<FilterEffectApplier> createSoftwareApplier() const = 0;
    virtual std::optional<GraphicsStyle> createGraphicsStyle(GraphicsContext&, const Filter&) const { return std::nullopt; }

    RefPtr<FilterImage> apply(const Filter&, FilterImage& input, FilterResults&) override;
    FilterStyleVector createFilterStyles(GraphicsContext&, const Filter&, const FilterStyle& input) const override;

    DestinationColorSpace m_operatingColorSpace { DestinationColorSpace::SRGB() };
};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const FilterEffect&);

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::FilterEffect)
    static bool isType(const WebCore::FilterFunction& function) { return function.isFilterEffect(); }
SPECIALIZE_TYPE_TRAITS_END()
