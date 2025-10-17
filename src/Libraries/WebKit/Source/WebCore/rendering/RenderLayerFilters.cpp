/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 12, 2024.
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
#include "config.h"
#include "RenderLayerFilters.h"

#include "CSSFilter.h"
#include "CachedSVGDocument.h"
#include "CachedSVGDocumentReference.h"
#include "GraphicsContextSwitcher.h"
#include "LegacyRenderSVGResourceFilter.h"
#include "Logging.h"
#include "RenderSVGShape.h"
#include "RenderStyleInlines.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RenderLayerFilters);

RenderLayerFilters::RenderLayerFilters(RenderLayer& layer)
    : m_layer(layer)
{
}

RenderLayerFilters::~RenderLayerFilters()
{
    removeReferenceFilterClients();
}

bool RenderLayerFilters::hasFilterThatMovesPixels() const
{
    return m_filter && m_filter->hasFilterThatMovesPixels();
}

bool RenderLayerFilters::hasFilterThatShouldBeRestrictedBySecurityOrigin() const
{
    return m_filter && m_filter->hasFilterThatShouldBeRestrictedBySecurityOrigin();
}

bool RenderLayerFilters::hasSourceImage() const
{
    return m_targetSwitcher && m_targetSwitcher->hasSourceImage();
}

void RenderLayerFilters::notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess)
{
    // FIXME: This really shouldn't have to invalidate layer composition,
    // but tests like css3/filters/effect-reference-delete.html fail if that doesn't happen.
    if (auto* enclosingElement = m_layer.enclosingElement())
        enclosingElement->invalidateStyleAndLayerComposition();
    m_layer.renderer().repaint();
}

void RenderLayerFilters::updateReferenceFilterClients(const FilterOperations& operations)
{
    removeReferenceFilterClients();

    for (auto& operation : operations) {
        RefPtr referenceOperation = dynamicDowncast<ReferenceFilterOperation>(operation);
        if (!referenceOperation)
            continue;

        auto* documentReference = referenceOperation->cachedSVGDocumentReference();
        if (auto* cachedSVGDocument = documentReference ? documentReference->document() : nullptr) {
            // Reference is external; wait for notifyFinished().
            cachedSVGDocument->addClient(*this);
            m_externalSVGReferences.append(cachedSVGDocument);
        } else {
            // Reference is internal; add layer as a client so we can trigger filter repaint on SVG attribute change.
            RefPtr filterElement = m_layer.renderer().document().getElementById(referenceOperation->fragment());
            if (!filterElement)
                continue;
            CheckedPtr renderer = dynamicDowncast<LegacyRenderSVGResourceFilter>(filterElement->renderer());
            if (!renderer)
                continue;
            renderer->addClientRenderLayer(m_layer);
            m_internalSVGReferences.append(WTFMove(filterElement));
        }
    }
}

void RenderLayerFilters::removeReferenceFilterClients()
{
    for (auto& resourceHandle : m_externalSVGReferences)
        resourceHandle->removeClient(*this);

    m_externalSVGReferences.clear();

    for (auto& filterElement : m_internalSVGReferences) {
        if (auto* renderer = filterElement->renderer())
            downcast<LegacyRenderSVGResourceContainer>(*renderer).removeClientRenderLayer(m_layer);
    }
    m_internalSVGReferences.clear();
}

bool RenderLayerFilters::isIdentity(RenderElement& renderer)
{
    const auto& operations = renderer.style().filter();
    return CSSFilter::isIdentity(renderer, operations);
}

IntOutsets RenderLayerFilters::calculateOutsets(RenderElement& renderer, const FloatRect& targetBoundingBox)
{
    const auto& operations = renderer.style().filter();
    
    if (!operations.hasFilterThatMovesPixels())
        return { };

    return CSSFilter::calculateOutsets(renderer, operations, targetBoundingBox);
}

GraphicsContext* RenderLayerFilters::beginFilterEffect(RenderElement& renderer, GraphicsContext& context, const LayoutRect& filterBoxRect, const LayoutRect& dirtyRect, const LayoutRect& layerRepaintRect, const LayoutRect& clipRect)
{
    auto expandedDirtyRect = dirtyRect;
    auto targetBoundingBox = intersection(filterBoxRect, dirtyRect);

    auto outsets = calculateOutsets(renderer, targetBoundingBox);
    if (!outsets.isZero()) {
        LayoutBoxExtent flippedOutsets { outsets.bottom(), outsets.left(), outsets.top(), outsets.right() };
        expandedDirtyRect.expand(flippedOutsets);
    }

    if (is<RenderSVGShape>(renderer))
        targetBoundingBox = enclosingLayoutRect(renderer.objectBoundingBox());
    else {
        // Calculate targetBoundingBox since it will be used if the filter is created.
        targetBoundingBox = intersection(filterBoxRect, expandedDirtyRect);
    }

    if (targetBoundingBox.isEmpty())
        return nullptr;

    if (!m_filter || m_targetBoundingBox != targetBoundingBox) {
        m_targetBoundingBox = targetBoundingBox;
        // FIXME: This rebuilds the entire effects chain even if the filter style didn't change.
        m_filter = CSSFilter::create(renderer, renderer.style().filter(), m_preferredFilterRenderingModes, m_filterScale, m_targetBoundingBox, context);
    }

    if (!m_filter)
        return nullptr;

    auto& filter = *m_filter;
    auto filterRegion = m_targetBoundingBox;

    if (filter.hasFilterThatMovesPixels()) {
        // For CSSFilter, filterRegion = targetBoundingBox + filter->outsets()
        filterRegion.expand(toLayoutBoxExtent(outsets));
    } else if (auto* shape = dynamicDowncast<RenderSVGShape>(renderer))
        filterRegion = shape->currentSVGLayoutRect();

    if (filterRegion.isEmpty())
        return nullptr;

    // For CSSFilter, sourceImageRect = filterRegion.
    bool hasUpdatedBackingStore = false;
    if (m_filterRegion != filterRegion) {
        m_filterRegion = filterRegion;
        hasUpdatedBackingStore = true;
    }

    filter.setFilterRegion(m_filterRegion);

    if (!filter.hasFilterThatMovesPixels())
        m_repaintRect = dirtyRect;
    else if (hasUpdatedBackingStore || !hasSourceImage())
        m_repaintRect = filterRegion;
    else {
        m_repaintRect = dirtyRect;
        m_repaintRect.unite(layerRepaintRect);
        m_repaintRect.intersect(filterRegion);
    }

    resetDirtySourceRect();

    if (!m_targetSwitcher || hasUpdatedBackingStore) {
        FloatRect sourceImageRect;
        if (is<RenderSVGShape>(renderer))
            sourceImageRect = renderer.strokeBoundingBox();
        else
            sourceImageRect = m_targetBoundingBox;
        m_targetSwitcher = GraphicsContextSwitcher::create(context, sourceImageRect, DestinationColorSpace::SRGB(), { &filter });
    }

    if (!m_targetSwitcher)
        return nullptr;

    m_targetSwitcher->beginClipAndDrawSourceImage(context, m_repaintRect, clipRect);

    return m_targetSwitcher->drawingContext(context);
}

void RenderLayerFilters::applyFilterEffect(GraphicsContext& destinationContext)
{
    LOG_WITH_STREAM(Filters, stream << "\nRenderLayerFilters " << this << " applyFilterEffect");

    ASSERT(m_targetSwitcher);
    m_targetSwitcher->endClipAndDrawSourceImage(destinationContext, DestinationColorSpace::SRGB());

    LOG_WITH_STREAM(Filters, stream << "RenderLayerFilters " << this << " applyFilterEffect done\n");
}

} // namespace WebCore
