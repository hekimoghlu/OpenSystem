/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 13, 2025.
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
#include "PDFDataDetectorOverlayController.h"

#if ENABLE(UNIFIED_PDF_DATA_DETECTION)

#include "PDFDataDetectorItem.h"
#include "PDFKitSPI.h"
#include "UnifiedPDFPlugin.h"
#include "WebMouseEvent.h"
#include <WebCore/DataDetectorElementInfo.h>
#include <WebCore/FloatRect.h>
#include <WebCore/FrameView.h>
#include <WebCore/GraphicsLayer.h>
#include <WebCore/GraphicsLayerClient.h>
#include <WebCore/IntPoint.h>
#include <WebCore/Page.h>
#include <pal/spi/cocoa/DataDetectorsCoreSPI.h>
#include <pal/spi/mac/DataDetectorsSPI.h>
#include <wtf/Algorithms.h>
#include <wtf/IteratorRange.h>
#include <wtf/OptionSet.h>
#include <wtf/RefPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/Scope.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>
#include <wtf/cocoa/VectorCocoa.h>
#include <wtf/text/WTFString.h>

#include <pal/mac/DataDetectorsSoftLink.h>

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(PDFDataDetectorOverlayController);

PDFDataDetectorOverlayController::PDFDataDetectorOverlayController(UnifiedPDFPlugin& plugin)
    : m_plugin(plugin)
{
}

PDFDataDetectorOverlayController::~PDFDataDetectorOverlayController() = default;

RefPtr<UnifiedPDFPlugin> PDFDataDetectorOverlayController::protectedPlugin() const
{
    return m_plugin.get();
}

PageOverlay& PDFDataDetectorOverlayController::installOverlayIfNeeded()
{
    if (m_overlay)
        return *m_overlay;

    m_overlay = PageOverlay::create(*this, PageOverlay::OverlayType::Document);
    protectedPlugin()->installDataDetectorOverlay(Ref { *m_overlay });

    return *m_overlay;
}

void PDFDataDetectorOverlayController::uninstallOverlay()
{
    if (!m_overlay)
        return;

    RefPtr plugin = protectedPlugin();
    if (!plugin)
        return;

    plugin->uninstallDataDetectorOverlay(Ref { *std::exchange(m_overlay, nullptr) });
}

void PDFDataDetectorOverlayController::teardown()
{
    uninstallOverlay();
}

static RetainPtr<DDHighlightRef> createPlatformDataDetectorHighlight(Vector<FloatRect>&& highlightBounds, FloatRect&& visibleContentRect)
{
    Vector highlightBoundsCG = highlightBounds.map([](const FloatRect& bound) -> CGRect {
        return bound;
    });

    DDHighlightStyle style = static_cast<DDHighlightStyle>(DDHighlightStyleBubbleStandard) | static_cast<DDHighlightStyle>(DDHighlightStyleStandardIconArrow);
    BOOL drawButton = YES;
    NSWritingDirection writingDirection = NSWritingDirectionNatural;
    BOOL endsWithEOL = NO;
    BOOL drawFlipped = YES;
    float targetSurfaceBackingScaleFactor = 0;

    return adoptCF(PAL::softLink_DataDetectors_DDHighlightCreateWithRectsInVisibleRectWithStyleScaleAndDirection(nullptr, highlightBoundsCG.data(), highlightBounds.size(), visibleContentRect, style, drawButton, writingDirection, endsWithEOL, drawFlipped, targetSurfaceBackingScaleFactor));
}

RetainPtr<DDHighlightRef> PDFDataDetectorOverlayController::createPlatformDataDetectorHighlight(PDFDataDetectorItem& dataDetectorItem) const
{
    RefPtr plugin = protectedPlugin();
    if (!plugin)
        return { };

    RefPtr mainFrameView = plugin->mainFrameView();
    if (!mainFrameView)
        return { };

    auto rectForSelectionInMainFrameContentsSpace = plugin->rectForSelectionInMainFrameContentsSpace(dataDetectorItem.selection().get());

    return ::WebKit::createPlatformDataDetectorHighlight(Vector<FloatRect>::from(WTFMove(rectForSelectionInMainFrameContentsSpace)), mainFrameView->visibleContentRect());
}

bool PDFDataDetectorOverlayController::handleMouseEvent(const WebMouseEvent& event, PDFDocumentLayout::PageIndex pageIndex)
{
    RefPtr plugin = protectedPlugin();
    if (!plugin)
        return false;

    if (plugin->isLocked())
        return false;

    RefPtr mainFrameView = plugin->mainFrameView();
    if (!mainFrameView)
        return false;

    if (!PAL::isDataDetectorsFrameworkAvailable())
        return false;

    updateDataDetectorHighlightsIfNeeded(pageIndex);

    auto mousePositionInWindowSpace = event.position();
    auto mousePositionInMainFrameContentsSpace = mainFrameView->windowToContents(mousePositionInWindowSpace);
    bool mouseIsOverActiveHighlightButton = false;
    m_staleDataDetectorItemWithHighlight = std::exchange(m_activeDataDetectorItemWithHighlight, { { }, { } });
    RefPtr<PDFDataDetectorItem> dataDetectorItemForActiveHighlight;

    if (auto iterator = m_pdfDataDetectorItemsWithHighlightsMap.find(pageIndex); iterator != m_pdfDataDetectorItemsWithHighlightsMap.end()) {
        for (auto& [dataDetectorItem, coreHighlight] : iterator->value) {
            Boolean isOverButton = NO;
            if (!PAL::softLink_DataDetectors_DDHighlightPointIsOnHighlight(coreHighlight->highlight(), mousePositionInMainFrameContentsSpace, &isOverButton))
                continue;

            mouseIsOverActiveHighlightButton = isOverButton;
            m_activeDataDetectorItemWithHighlight.first = dataDetectorItem.copyRef();
            m_activeDataDetectorItemWithHighlight.second = coreHighlight.copyRef();
            break;
        }
    }

    RefPtr previousActiveHighlight = m_staleDataDetectorItemWithHighlight.second;
    RefPtr activeHighlight = m_activeDataDetectorItemWithHighlight.second;

    if (previousActiveHighlight != activeHighlight) {
        if (previousActiveHighlight)
            previousActiveHighlight->fadeOut();

        if (activeHighlight) {
            installOverlayIfNeeded().layer().addChild(activeHighlight->protectedLayer());
            activeHighlight->fadeIn();
        }

        didInvalidateHighlightOverlayRects(pageIndex, ShouldUpdatePlatformHighlightData::No, ActiveHighlightChanged::Yes);
    }

    if (event.type() == WebEventType::MouseDown && mouseIsOverActiveHighlightButton)
        return handleDataDetectorAction(mousePositionInWindowSpace, Ref { *m_activeDataDetectorItemWithHighlight.first });

    return false;
}

bool PDFDataDetectorOverlayController::handleDataDetectorAction(const IntPoint& mousePositionInWindowSpace, PDFDataDetectorItem& dataDetectorItem)
{
    RefPtr plugin = protectedPlugin();
    if (!plugin)
        return false;

    RefPtr mainFrameView = plugin->mainFrameView();
    if (!mainFrameView)
        return false;

    auto rectForSelectionInMainFrameContentsSpace = roundedIntRect(plugin->rectForSelectionInMainFrameContentsSpace(dataDetectorItem.selection().get()));

    plugin->handleClickForDataDetectionResult({ dataDetectorItem.scannerResult(), mainFrameView->contentsToWindow(rectForSelectionInMainFrameContentsSpace) }, mousePositionInWindowSpace);
    return true;
}

void PDFDataDetectorOverlayController::updateDataDetectorHighlightsIfNeeded(PDFDocumentLayout::PageIndex pageIndex)
{
    RefPtr plugin = protectedPlugin();
    if (!plugin)
        return;

    if (!PAL::isDataDetectorsFrameworkAvailable())
        return;

    bool shouldUpdatePlatformHighlightData = true;

    // FIXME: Simplify this logic to only cache on the first PDFPage query by calling `.ensure()` once we figure out why the first few fetches sometimes return bogus results.
    if (auto it = m_pdfDataDetectorItemsWithHighlightsMap.find(pageIndex); it == m_pdfDataDetectorItemsWithHighlightsMap.end() || it->value.isEmpty()) {
        auto dataDetectorItemsWithHighlights = [&]() -> PDFDataDetectorItemsWithHighlights {
            RetainPtr page = plugin->pageAtIndex(pageIndex);
#if HAVE(PDFPAGE_DATA_DETECTOR_RESULTS)
            if ([page respondsToSelector:@selector(dataDetectorResults)]) {
                RetainPtr<NSArray> results = [page dataDetectorResults];
                return makeVector(results.get(), [&](DDScannerResult *result) -> std::optional<PDFDataDetectorItemWithHighlight> {
                    Ref dataDetectorItem = PDFDataDetectorItem::create(result, page.get());
                    if (!dataDetectorItem->hasActions() || dataDetectorItem->isPastDate())
                        return { };

                    Ref coreHighlight = DataDetectorHighlight::createForPDFSelection(*this, createPlatformDataDetectorHighlight(dataDetectorItem.get()));

                    return { std::make_pair(WTFMove(dataDetectorItem), WTFMove(coreHighlight)) };
                });
            }
#endif
            return { };
        }();

        m_pdfDataDetectorItemsWithHighlightsMap.set(pageIndex, WTFMove(dataDetectorItemsWithHighlights));
        shouldUpdatePlatformHighlightData = false;
    }

    if (shouldUpdatePlatformHighlightData)
        updatePlatformHighlightData(pageIndex);
}

void PDFDataDetectorOverlayController::updatePlatformHighlightData(PDFDocumentLayout::PageIndex pageIndex)
{
    for (auto& [dataDetectorItem, coreHighlight] : m_pdfDataDetectorItemsWithHighlightsMap.get(pageIndex))
        coreHighlight->setHighlight(createPlatformDataDetectorHighlight(Ref { dataDetectorItem }).get());
}

void PDFDataDetectorOverlayController::hideActiveHighlightOverlay()
{
    if (RefPtr activeHighlight = m_activeDataDetectorItemWithHighlight.second)
        activeHighlight->dismissImmediately();
}

void PDFDataDetectorOverlayController::didInvalidateHighlightOverlayRects(std::optional<PDFDocumentLayout::PageIndex> pageIndex, ShouldUpdatePlatformHighlightData shouldUpdatePlatformHighlightData, ActiveHighlightChanged activeHighlightChanged)
{
    // Regardless of what we repaint, we don't need the stale data after this.
    auto resetStaleDataDetectorWithHighlight = makeScopeExit([&] {
        m_staleDataDetectorItemWithHighlight = { { }, { } };
    });

    RefPtr plugin = protectedPlugin();
    if (!plugin)
        return;

    auto [previousDataDetectorItem, previousActiveHighlight] = m_staleDataDetectorItemWithHighlight;
    auto [activeDataDetectorItem, activeHighlight] = m_activeDataDetectorItemWithHighlight;

    bool shouldUpdateHighlights = activeHighlightChanged == ActiveHighlightChanged::Yes || activeHighlight || previousActiveHighlight;
    if (!shouldUpdateHighlights || !plugin->canShowDataDetectorHighlightOverlays())
        return;

    if (shouldUpdatePlatformHighlightData == ShouldUpdatePlatformHighlightData::No)
        return;

    auto pageIndices = [&] {
        if (pageIndex) {
            if (auto iterator = m_pdfDataDetectorItemsWithHighlightsMap.find(*pageIndex); iterator != m_pdfDataDetectorItemsWithHighlightsMap.end())
                return makeSizedIteratorRange(m_pdfDataDetectorItemsWithHighlightsMap, iterator.keys(), std::next(iterator).keys());
        }

        return m_pdfDataDetectorItemsWithHighlightsMap.keys();
    }();

    for (auto pageIndex : pageIndices)
        updatePlatformHighlightData(pageIndex);
}

#pragma mark - PageOverlayClient

void PDFDataDetectorOverlayController::willMoveToPage(PageOverlay&, Page* page)
{
    if (!page)
        uninstallOverlay();
}

#pragma mark - DataDetectorHighlightClient

void PDFDataDetectorOverlayController::scheduleRenderingUpdate(OptionSet<RenderingUpdateStep> requestedSteps)
{
    RefPtr plugin = protectedPlugin();
    if (!plugin)
        return;

    plugin->scheduleRenderingUpdate(requestedSteps);
}

float PDFDataDetectorOverlayController::deviceScaleFactor() const
{
    RefPtr plugin = protectedPlugin();
    if (!plugin)
        return 1;

    return plugin->deviceScaleFactor();
}

RefPtr<GraphicsLayer> PDFDataDetectorOverlayController::createGraphicsLayer(GraphicsLayerClient& client)
{
    RefPtr plugin = protectedPlugin();
    if (!plugin)
        return nullptr;

    return plugin->createGraphicsLayer(client);
}

} // namespace WebKit

#endif // ENABLE(UNIFIED_PDF_DATA_DETECTION)

