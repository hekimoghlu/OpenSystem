/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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
#include "PDFPresentationController.h"

#if ENABLE(UNIFIED_PDF)

#include "AsyncPDFRenderer.h"
#include "PDFDiscretePresentationController.h"
#include "PDFKitSPI.h"
#include "PDFScrollingPresentationController.h"
#include <WebCore/GraphicsLayer.h>
#include <wtf/CheckedRef.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(PDFPresentationController);

RefPtr<PDFPresentationController> PDFPresentationController::createForMode(PDFDocumentLayout::DisplayMode mode, UnifiedPDFPlugin& plugin)
{
    if (PDFDocumentLayout::isScrollingDisplayMode(mode))
        return adoptRef(*new PDFScrollingPresentationController { plugin });

    if (PDFDocumentLayout::isDiscreteDisplayMode(mode))
        return adoptRef(*new PDFDiscretePresentationController { plugin });

    ASSERT_NOT_REACHED();
    return nullptr;
}

PDFPresentationController::PDFPresentationController(UnifiedPDFPlugin& plugin)
    : m_plugin(plugin)
{

}

PDFPresentationController::~PDFPresentationController() = default;

void PDFPresentationController::teardown()
{
    clearAsyncRenderer();
}

Ref<AsyncPDFRenderer> PDFPresentationController::asyncRenderer()
{
    if (m_asyncRenderer)
        return *m_asyncRenderer;

    m_asyncRenderer = AsyncPDFRenderer::create(*this);
    return *m_asyncRenderer;
}

RefPtr<AsyncPDFRenderer> PDFPresentationController::asyncRendererIfExists() const
{
    return m_asyncRenderer;
}

void PDFPresentationController::clearAsyncRenderer()
{
    if (RefPtr asyncRenderer = std::exchange(m_asyncRenderer, nullptr))
        asyncRenderer->teardown();
}

RefPtr<GraphicsLayer> PDFPresentationController::createGraphicsLayer(const String& name, GraphicsLayer::Type layerType)
{
    auto* graphicsLayerFactory = m_plugin->graphicsLayerFactory();
    Ref graphicsLayer = GraphicsLayer::create(graphicsLayerFactory, graphicsLayerClient(), layerType);
    graphicsLayer->setName(name);
    return graphicsLayer;
}

RefPtr<GraphicsLayer> PDFPresentationController::makePageContainerLayer(PDFDocumentLayout::PageIndex pageIndex)
{
    auto addLayerShadow = [](GraphicsLayer& layer, IntPoint shadowOffset, const Color& shadowColor, int shadowStdDeviation) {
        Vector<Ref<FilterOperation>> filterOperations;
        filterOperations.append(DropShadowFilterOperation::create(shadowOffset, shadowStdDeviation, shadowColor));
        layer.setFilters(FilterOperations { WTFMove(filterOperations) });
    };

    constexpr auto containerShadowOffset = IntPoint { 0, 1 };
    constexpr auto containerShadowColor = SRGBA<uint8_t> { 0, 0, 0, 46 };
    constexpr int containerShadowStdDeviation = 2;

    constexpr auto shadowOffset = IntPoint { 0, 2 };
    constexpr auto shadowColor = SRGBA<uint8_t> { 0, 0, 0, 38 };
    constexpr int shadowStdDeviation = 6;

    RefPtr pageContainerLayer = createGraphicsLayer(makeString("Page container "_s, pageIndex), GraphicsLayer::Type::Normal);
    RefPtr pageBackgroundLayer = createGraphicsLayer(makeString("Page background "_s, pageIndex), GraphicsLayer::Type::Normal);
    // Can only be null if this->page() is null, which we checked above.
    ASSERT(pageContainerLayer);
    ASSERT(pageBackgroundLayer);

    pageContainerLayer->setAnchorPoint({ });
    addLayerShadow(*pageContainerLayer, containerShadowOffset, containerShadowColor, containerShadowStdDeviation);

    pageBackgroundLayer->setAnchorPoint({ });
    pageBackgroundLayer->setBackgroundColor(Color::white);

    pageBackgroundLayer->setDrawsContent(true);
    pageBackgroundLayer->setAcceleratesDrawing(true);
    pageBackgroundLayer->setShouldUpdateRootRelativeScaleFactor(false);
    pageBackgroundLayer->setAllowsTiling(false);
    pageBackgroundLayer->setNeedsDisplay(); // We only need to paint this layer once when page backgrounds change.

    // FIXME: <https://webkit.org/b/276981> Need to add a 1px black border with alpha 0.0586.

    addLayerShadow(*pageBackgroundLayer, shadowOffset, shadowColor, shadowStdDeviation);

    pageContainerLayer->addChild(*pageBackgroundLayer);

    return pageContainerLayer;
}

RefPtr<GraphicsLayer> PDFPresentationController::pageBackgroundLayerForPageContainerLayer(GraphicsLayer& pageContainerLayer)
{
    auto& children = pageContainerLayer.children();
    if (children.size()) {
        Ref layer = children[0];
        return WTFMove(layer);
    }

    return nullptr;
}

void PDFPresentationController::releaseMemory()
{
    if (RefPtr asyncRenderer = asyncRendererIfExists())
        asyncRenderer->releaseMemory();
}

RetainPtr<PDFDocument> PDFPresentationController::pluginPDFDocument() const
{
    return m_plugin->pdfDocument();
}

FloatRect PDFPresentationController::layoutBoundsForPageAtIndex(PDFDocumentLayout::PageIndex pageIndex) const
{
    return m_plugin->layoutBoundsForPageAtIndex(pageIndex);
}

bool PDFPresentationController::pluginShouldCachePagePreviews() const
{
    return m_plugin->shouldCachePagePreviews();
}

float PDFPresentationController::scaleForPagePreviews() const
{
    return m_plugin->scaleForPagePreviews();
}

void PDFPresentationController::setNeedsRepaintForPageCoverage(RepaintRequirements repaintRequirements, const PDFPageCoverage& coverage)
{
    // HoverOverlay is currently painted to PDFContent.
    if (repaintRequirements.contains(RepaintRequirement::HoverOverlay)) {
        repaintRequirements.remove(RepaintRequirement::HoverOverlay);
        repaintRequirements.add(RepaintRequirement::PDFContent);
    }

    auto layerCoverages = layerCoveragesForRepaintPageCoverage(repaintRequirements, coverage);
    for (auto& layerCoverage : layerCoverages)
        Ref { layerCoverage.layer }->setNeedsDisplayInRect(layerCoverage.bounds);

    // Unite consecutive PDFContent display rects and send them as render rect to AsyncRenderer.
    if (RefPtr asyncRenderer = asyncRendererIfExists()) {
        RefPtr<GraphicsLayer> layer;
        FloatRect bounds;
        for (auto& layerCoverage : layerCoverages) {
            if (!layerCoverage.repaintRequirements.contains(RepaintRequirement::PDFContent))
                continue;
            if (layerCoverage.layer.ptr() != layer) {
                if (layer && !bounds.isEmpty())
                    asyncRenderer->setNeedsRenderForRect(*layer, bounds);
                layer = layerCoverage.layer.ptr();
                bounds = layerCoverage.bounds;
            } else
                bounds.unite(layerCoverage.bounds);
        }
        if (layer && !bounds.isEmpty())
            asyncRenderer->setNeedsRenderForRect(*layer, bounds);
        asyncRenderer->setNeedsPagePreviewRenderForPageCoverage(coverage);
    }
}

PDFDocumentLayout::PageIndex PDFPresentationController::nearestPageIndexForDocumentPoint(const FloatPoint& point) const
{
    if (m_plugin->isLocked())
        return 0;
    return m_plugin->documentLayout().nearestPageIndexForDocumentPoint(point, visibleRow());
}

std::optional<PDFDocumentLayout::PageIndex> PDFPresentationController::pageIndexForDocumentPoint(const FloatPoint& point) const
{
    auto& documentLayout = m_plugin->documentLayout();

    if (auto row = visibleRow()) {
        for (auto pageIndex : row->pages) {
            auto pageBounds = documentLayout.layoutBoundsForPageAtIndex(pageIndex);
            if (pageBounds.contains(point))
                return pageIndex;
        }

        return { };
    }

    for (PDFDocumentLayout::PageIndex pageIndex = 0; pageIndex < documentLayout.pageCount(); ++pageIndex) {
        auto pageBounds = documentLayout.layoutBoundsForPageAtIndex(pageIndex);
        if (pageBounds.contains(point))
            return pageIndex;
    }

    return { };
}

auto PDFPresentationController::pdfPositionForCurrentView(AnchorPoint anchorPoint, bool preservePosition) const -> std::optional<VisiblePDFPosition>
{
    if (!preservePosition)
        return { };

    auto& documentLayout = m_plugin->documentLayout();
    if (!documentLayout.hasLaidOutPDFDocument())
        return { };

    auto maybePageIndex = pageIndexForCurrentView(anchorPoint);
    if (!maybePageIndex)
        return { };

    auto pageIndex = *maybePageIndex;
    auto pageBounds = documentLayout.layoutBoundsForPageAtIndex(pageIndex);
    auto topLeftInDocumentSpace = m_plugin->convertDown(UnifiedPDFPlugin::CoordinateSpace::Plugin, UnifiedPDFPlugin::CoordinateSpace::PDFDocumentLayout, FloatPoint { });
    auto pagePoint = documentLayout.documentToPDFPage(FloatPoint { pageBounds.center().x(), topLeftInDocumentSpace.y() }, pageIndex);

    LOG_WITH_STREAM(PDF, stream << "PDFPresentationController::pdfPositionForCurrentView - point " << pagePoint << " in page " << pageIndex << " with anchor point " << std::to_underlying(anchorPoint));

    return VisiblePDFPosition { pageIndex, pagePoint };
}

FloatPoint PDFPresentationController::anchorPointInDocumentSpace(AnchorPoint anchorPoint) const
{
    auto anchorPointInPluginSpace = [anchorPoint, checkedPlugin = CheckedRef { m_plugin.get() }] -> FloatPoint {
        switch (anchorPoint) {
        case AnchorPoint::TopLeft:
            return { };
        case AnchorPoint::Center:
            return flooredIntPoint(checkedPlugin->size() / 2);
        }
        ASSERT_NOT_REACHED();
        return { };
    }();
    return m_plugin->convertDown(UnifiedPDFPlugin::CoordinateSpace::Plugin, UnifiedPDFPlugin::CoordinateSpace::PDFDocumentLayout, anchorPointInPluginSpace);
}

} // namespace WebKit

#endif // ENABLE(UNIFIED_PDF)
