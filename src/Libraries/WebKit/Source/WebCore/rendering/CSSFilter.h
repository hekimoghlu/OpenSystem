/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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
#include "LengthBox.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FilterOperations;
class GraphicsContext;
class RenderElement;

class CSSFilter final : public Filter {
    WTF_MAKE_TZONE_ALLOCATED(CSSFilter);
public:
    static RefPtr<CSSFilter> create(RenderElement&, const FilterOperations&, OptionSet<FilterRenderingMode> preferredFilterRenderingModes, const FloatSize& filterScale, const FloatRect& targetBoundingBox, const GraphicsContext& destinationContext);
    WEBCORE_EXPORT static Ref<CSSFilter> create(Vector<Ref<FilterFunction>>&&);
    WEBCORE_EXPORT static Ref<CSSFilter> create(Vector<Ref<FilterFunction>>&&, OptionSet<FilterRenderingMode>, const FloatSize& filterScale, const FloatRect& filterRegion);

    const Vector<Ref<FilterFunction>>& functions() const { return m_functions; }

    void setFilterRegion(const FloatRect&);

    bool hasFilterThatMovesPixels() const { return m_hasFilterThatMovesPixels; }
    bool hasFilterThatShouldBeRestrictedBySecurityOrigin() const { return m_hasFilterThatShouldBeRestrictedBySecurityOrigin; }

    FilterEffectVector effectsOfType(FilterFunction::Type) const final;

    RefPtr<FilterImage> apply(FilterImage* sourceImage, FilterResults&) final;
    FilterStyleVector createFilterStyles(GraphicsContext&, const FilterStyle& sourceStyle) const final;

    static bool isIdentity(RenderElement&, const FilterOperations&);
    static IntOutsets calculateOutsets(RenderElement&, const FilterOperations&, const FloatRect& targetBoundingBox);

private:
    CSSFilter(const FloatSize& filterScale, bool hasFilterThatMovesPixels, bool hasFilterThatShouldBeRestrictedBySecurityOrigin);
    CSSFilter(Vector<Ref<FilterFunction>>&&);
    CSSFilter(Vector<Ref<FilterFunction>>&&, const FloatSize& filterScale, const FloatRect& filterRegion);

    bool buildFilterFunctions(RenderElement&, const FilterOperations&, OptionSet<FilterRenderingMode> preferredFilterRenderingModes, const FloatRect& targetBoundingBox, const GraphicsContext& destinationContext);

    OptionSet<FilterRenderingMode> supportedFilterRenderingModes() const final;

    WTF::TextStream& externalRepresentation(WTF::TextStream&, FilterRepresentation) const final;

    bool m_hasFilterThatMovesPixels { false };
    bool m_hasFilterThatShouldBeRestrictedBySecurityOrigin { false };

    Vector<Ref<FilterFunction>> m_functions;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_FILTER_FUNCTION(CSSFilter);
