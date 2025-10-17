/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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

#include "FilterResults.h"
#include "GraphicsContextSwitcher.h"
#include "LegacyRenderSVGResourceContainer.h"
#include "SVGFilter.h"
#include "SVGUnitTypes.h"
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class GraphicsContext;
class SVGFilterElement;

struct FilterData {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(FilterData);
    WTF_MAKE_NONCOPYABLE(FilterData);
public:
    enum FilterDataState { PaintingSource, Applying, Built, CycleDetected, MarkedForRemoval };

    FilterData() = default;

    RefPtr<SVGFilter> filter;

    std::unique_ptr<GraphicsContextSwitcher> targetSwitcher;
    FloatRect sourceImageRect;

    GraphicsContext* savedContext { nullptr };
    FilterDataState state { PaintingSource };
};

class LegacyRenderSVGResourceFilter final : public LegacyRenderSVGResourceContainer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGResourceFilter);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGResourceFilter);
public:
    LegacyRenderSVGResourceFilter(SVGFilterElement&, RenderStyle&&);
    virtual ~LegacyRenderSVGResourceFilter();

    inline SVGFilterElement& filterElement() const;
    inline Ref<SVGFilterElement> protectedFilterElement() const;
    bool isIdentity() const;

    void removeAllClientsFromCache() override;
    void removeClientFromCache(RenderElement&) override;

    OptionSet<ApplyResult> applyResource(RenderElement&, const RenderStyle&, GraphicsContext*&, OptionSet<RenderSVGResourceMode>) override;
    void postApplyResource(RenderElement&, GraphicsContext*&, OptionSet<RenderSVGResourceMode>, const Path*, const RenderElement*) override;

    FloatRect resourceBoundingBox(const RenderObject&, RepaintRectCalculation) override;

    inline SVGUnitTypes::SVGUnitType filterUnits() const;
    inline SVGUnitTypes::SVGUnitType primitiveUnits() const;

    void markFilterForRepaint(FilterEffect&);
    void markFilterForRebuild();

    RenderSVGResourceType resourceType() const override { return FilterResourceType; }

    FloatRect drawingRegion(RenderObject&) const;

private:
    void element() const = delete;

    ASCIILiteral renderName() const override { return "RenderSVGResourceFilter"_s; }

    UncheckedKeyHashMap<SingleThreadWeakRef<RenderObject>, std::unique_ptr<FilterData>> m_rendererFilterDataMap;
};

WTF::TextStream& operator<<(WTF::TextStream&, FilterData::FilterDataState);

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::LegacyRenderSVGResourceFilter)
    static bool isType(const WebCore::RenderObject& renderer) { return renderer.isLegacyRenderSVGResourceFilter(); }
    static bool isType(const WebCore::LegacyRenderSVGResource& resource) { return resource.resourceType() == WebCore::FilterResourceType; }
SPECIALIZE_TYPE_TRAITS_END()
