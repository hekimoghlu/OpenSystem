/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 6, 2025.
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

#include "CachedResourceHandle.h"
#include "CachedSVGDocumentClient.h"
#include "FilterRenderingMode.h"
#include "RenderLayer.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class CachedSVGDocument;
class Element;
class FilterOperations;
class GraphicsContextSwitcher;

class RenderLayerFilters final : private CachedSVGDocumentClient {
    WTF_MAKE_TZONE_ALLOCATED(RenderLayerFilters);
public:
    explicit RenderLayerFilters(RenderLayer&);
    virtual ~RenderLayerFilters();

    const LayoutRect& dirtySourceRect() const { return m_dirtySourceRect; }
    void expandDirtySourceRect(const LayoutRect& rect) { m_dirtySourceRect.unite(rect); }

    CSSFilter* filter() const { return m_filter.get(); }
    void clearFilter() { m_filter = nullptr; }
    
    bool hasFilterThatMovesPixels() const;
    bool hasFilterThatShouldBeRestrictedBySecurityOrigin() const;
    bool hasSourceImage() const;

    void updateReferenceFilterClients(const FilterOperations&);
    void removeReferenceFilterClients();

    void setPreferredFilterRenderingModes(OptionSet<FilterRenderingMode> preferredFilterRenderingModes) { m_preferredFilterRenderingModes = preferredFilterRenderingModes; }
    void setFilterScale(const FloatSize& filterScale) { m_filterScale = filterScale; }

    static bool isIdentity(RenderElement&);
    static IntOutsets calculateOutsets(RenderElement&, const FloatRect& targetBoundingBox);

    // Per render
    LayoutRect repaintRect() const { return m_repaintRect; }

    GraphicsContext* beginFilterEffect(RenderElement&, GraphicsContext&, const LayoutRect& filterBoxRect, const LayoutRect& dirtyRect, const LayoutRect& layerRepaintRect, const LayoutRect& clipRect);
    void applyFilterEffect(GraphicsContext& destinationContext);

private:
    void notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess) final;
    void resetDirtySourceRect() { m_dirtySourceRect = LayoutRect(); }

    RenderLayer& m_layer;
    Vector<RefPtr<Element>> m_internalSVGReferences;
    Vector<CachedResourceHandle<CachedSVGDocument>> m_externalSVGReferences;

    LayoutRect m_targetBoundingBox;
    LayoutRect m_dirtySourceRect;
    LayoutRect m_repaintRect;

    OptionSet<FilterRenderingMode> m_preferredFilterRenderingModes { FilterRenderingMode::Software };
    FloatSize m_filterScale { 1, 1 };
    FloatRect m_filterRegion;

    RefPtr<CSSFilter> m_filter;
    std::unique_ptr<GraphicsContextSwitcher> m_targetSwitcher;
};

} // namespace WebCore
