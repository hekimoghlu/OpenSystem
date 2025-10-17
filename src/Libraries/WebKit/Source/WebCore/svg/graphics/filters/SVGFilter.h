/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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

#include "Filter.h"
#include "FilterResults.h"
#include "FloatRect.h"
#include "SVGFilterExpression.h"
#include "SVGUnitTypes.h"
#include <wtf/Ref.h>
#include <wtf/TypeCasts.h>

namespace WebCore {

class FilterImage;
class GraphicsContext;
class SVGFilterElement;

class SVGFilter final : public Filter {
public:
    static RefPtr<SVGFilter> create(SVGFilterElement&, OptionSet<FilterRenderingMode> preferredFilterRenderingModes, const FloatSize& filterScale, const FloatRect& filterRegion, const FloatRect& targetBoundingBox, const GraphicsContext& destinationContext, std::optional<RenderingResourceIdentifier> = std::nullopt);
    WEBCORE_EXPORT static Ref<SVGFilter> create(const FloatRect& targetBoundingBox, SVGUnitTypes::SVGUnitType primitiveUnits, SVGFilterExpression&&, FilterEffectVector&&, std::optional<RenderingResourceIdentifier>);
    WEBCORE_EXPORT static Ref<SVGFilter> create(const FloatRect& targetBoundingBox, SVGUnitTypes::SVGUnitType primitiveUnits, SVGFilterExpression&&, FilterEffectVector&&, std::optional<RenderingResourceIdentifier>, OptionSet<FilterRenderingMode>, const FloatSize& filterScale, const FloatRect& filterRegion);

    static bool isIdentity(SVGFilterElement&);
    static IntOutsets calculateOutsets(SVGFilterElement&, const FloatRect& targetBoundingBox);

    FloatRect targetBoundingBox() const { return m_targetBoundingBox; }
    SVGUnitTypes::SVGUnitType primitiveUnits() const { return m_primitiveUnits; }

    const SVGFilterExpression& expression() const { return m_expression; }
    const FilterEffectVector& effects() const { return m_effects; }

    FilterEffectVector effectsOfType(FilterFunction::Type) const final;

    WEBCORE_EXPORT FilterResults& ensureResults(const FilterResultsCreator&);
    void clearEffectResult(FilterEffect&);
    WEBCORE_EXPORT void mergeEffects(const FilterEffectVector&);

    RefPtr<FilterImage> apply(FilterImage* sourceImage, FilterResults&) final;
    FilterStyleVector createFilterStyles(GraphicsContext&, const FilterStyle& sourceStyle) const final;

    static FloatSize calculateResolvedSize(const FloatSize&, const FloatRect& targetBoundingBox, SVGUnitTypes::SVGUnitType primitiveUnits);

    WTF::TextStream& externalRepresentation(WTF::TextStream&, FilterRepresentation) const final;

private:
    SVGFilter(const FloatSize& filterScale, const FloatRect& filterRegion, const FloatRect& targetBoundingBox, SVGUnitTypes::SVGUnitType primitiveUnits, std::optional<RenderingResourceIdentifier>);
    SVGFilter(const FloatRect& targetBoundingBox, SVGUnitTypes::SVGUnitType primitiveUnits, SVGFilterExpression&&, FilterEffectVector&&, std::optional<RenderingResourceIdentifier>);
    SVGFilter(const FloatRect& targetBoundingBox, SVGUnitTypes::SVGUnitType primitiveUnits, SVGFilterExpression&&, FilterEffectVector&&, std::optional<RenderingResourceIdentifier>, const FloatSize& filterScale, const FloatRect& filterRegion);

    static std::optional<std::tuple<SVGFilterExpression, FilterEffectVector>> buildExpression(SVGFilterElement&, const SVGFilter&, const GraphicsContext& destinationContext);
    void setExpression(SVGFilterExpression&& expression) { m_expression = WTFMove(expression); }
    void setEffects(FilterEffectVector&& effects) { m_effects = WTFMove(effects); }

    FloatSize resolvedSize(const FloatSize&) const final;
    FloatPoint3D resolvedPoint3D(const FloatPoint3D&) const final;

    OptionSet<FilterRenderingMode> supportedFilterRenderingModes() const final;

    RefPtr<FilterImage> apply(const Filter&, FilterImage& sourceImage, FilterResults&) final;
    FilterStyleVector createFilterStyles(GraphicsContext&, const Filter&, const FilterStyle& sourceStyle) const final;

    FloatRect m_targetBoundingBox;
    SVGUnitTypes::SVGUnitType m_primitiveUnits;

    SVGFilterExpression m_expression;
    FilterEffectVector m_effects;

    std::unique_ptr<FilterResults> m_results;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_FILTER_FUNCTION(SVGFilter);
