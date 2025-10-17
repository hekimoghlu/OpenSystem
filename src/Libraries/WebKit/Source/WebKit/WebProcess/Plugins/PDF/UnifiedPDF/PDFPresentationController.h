/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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

#if ENABLE(UNIFIED_PDF)

#include "PDFDocumentLayout.h"
#include "PDFPageCoverage.h"
#include "UnifiedPDFPlugin.h"
#include <WebCore/GraphicsLayer.h>
#include <WebCore/PlatformLayerIdentifier.h>
#include <wtf/OptionSet.h>
#include <wtf/RefPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>

OBJC_CLASS PDFDocument;

namespace WebCore {
enum class TiledBackingScrollability : uint8_t;
class GraphicsLayerClient;
};

namespace WebKit {

class AsyncPDFRenderer;
class WebKeyboardEvent;
class UnifiedPDFPlugin;
class WebWheelEvent;
enum class RepaintRequirement : uint8_t;

class PDFPresentationController : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<PDFPresentationController> {
    WTF_MAKE_NONCOPYABLE(PDFPresentationController);
    WTF_MAKE_TZONE_ALLOCATED(PDFPresentationController);
public:
    static RefPtr<PDFPresentationController> createForMode(PDFDocumentLayout::DisplayMode, UnifiedPDFPlugin&);

    PDFPresentationController(UnifiedPDFPlugin&);
    virtual ~PDFPresentationController();

    virtual WebCore::GraphicsLayerClient& graphicsLayerClient() = 0;

    // Subclasses must call the base class teardown().
    virtual void teardown();

    virtual bool supportsDisplayMode(PDFDocumentLayout::DisplayMode) const = 0;
    virtual void willChangeDisplayMode(PDFDocumentLayout::DisplayMode newMode) = 0;

    // Package up the data needed to paint a set of pages for the given clip, for use by UnifiedPDFPlugin::paintPDFContent and async rendering.
    virtual PDFPageCoverage pageCoverageForContentsRect(const WebCore::FloatRect&, std::optional<PDFLayoutRow>) const = 0;
    virtual PDFPageCoverageAndScales pageCoverageAndScalesForContentsRect(const WebCore::FloatRect&, std::optional<PDFLayoutRow>, float tilingScaleFactor) const = 0;

    virtual WebCore::FloatRect convertFromContentsToPainting(const WebCore::FloatRect&, std::optional<PDFDocumentLayout::PageIndex> = { }) const = 0;
    virtual WebCore::FloatRect convertFromPaintingToContents(const WebCore::FloatRect&, std::optional<PDFDocumentLayout::PageIndex> = { }) const = 0;

    virtual void deviceOrPageScaleFactorChanged() = 0;

    virtual void setupLayers(WebCore::GraphicsLayer&) = 0;
    virtual void updateLayersOnLayoutChange(WebCore::FloatSize documentSize, WebCore::FloatSize centeringOffset, double scaleFactor) = 0;

    virtual void updateIsInWindow(bool isInWindow) = 0;
    virtual void updateDebugBorders(bool showDebugBorders, bool showRepaintCounters) = 0;

    virtual void updateForCurrentScrollability(OptionSet<WebCore::TiledBackingScrollability>) = 0;

    virtual void didGeneratePreviewForPage(PDFDocumentLayout::PageIndex) = 0;

    void setNeedsRepaintForPageCoverage(RepaintRequirements, const PDFPageCoverage&);

    virtual std::optional<PDFLayoutRow> visibleRow() const { return { }; }
    virtual std::optional<PDFLayoutRow> rowForLayer(const WebCore::GraphicsLayer*) const { return { }; }

    struct VisiblePDFPosition {
        PDFDocumentLayout::PageIndex pageIndex { 0 };
        WebCore::FloatPoint pagePoint;
    };

    enum class AnchorPoint : uint8_t { TopLeft, Center };
    std::optional<VisiblePDFPosition> pdfPositionForCurrentView(AnchorPoint, bool preservePosition = true) const;
    WebCore::FloatPoint anchorPointInDocumentSpace(AnchorPoint) const;
    virtual std::optional<PDFDocumentLayout::PageIndex> pageIndexForCurrentView(AnchorPoint) const = 0;
    virtual void restorePDFPosition(const VisiblePDFPosition&) = 0;

    virtual void ensurePageIsVisible(PDFDocumentLayout::PageIndex) = 0;

    WebCore::FloatRect layoutBoundsForPageAtIndex(PDFDocumentLayout::PageIndex) const;

    PDFDocumentLayout::PageIndex nearestPageIndexForDocumentPoint(const WebCore::FloatPoint&) const;
    std::optional<PDFDocumentLayout::PageIndex> pageIndexForDocumentPoint(const WebCore::FloatPoint&) const;

    // Event handling.
    virtual bool handleKeyboardEvent(const WebKeyboardEvent&) = 0;
    virtual bool wantsWheelEvents() const { return false; }
    virtual bool handleWheelEvent(const WebWheelEvent&) { return false; }

    void releaseMemory();
    RetainPtr<PDFDocument> pluginPDFDocument() const;
    bool pluginShouldCachePagePreviews() const;

    virtual std::optional<WebCore::PlatformLayerIdentifier> contentsLayerIdentifier() const { return std::nullopt; }

    float scaleForPagePreviews() const;
protected:
    RefPtr<WebCore::GraphicsLayer> createGraphicsLayer(const String&, WebCore::GraphicsLayer::Type);
    RefPtr<WebCore::GraphicsLayer> makePageContainerLayer(PDFDocumentLayout::PageIndex);
    struct LayerCoverage {
        Ref<WebCore::GraphicsLayer> layer;
        WebCore::FloatRect bounds;
        RepaintRequirements repaintRequirements;
    };
    virtual Vector<LayerCoverage> layerCoveragesForRepaintPageCoverage(RepaintRequirements, const PDFPageCoverage&) = 0;

    static RefPtr<WebCore::GraphicsLayer> pageBackgroundLayerForPageContainerLayer(WebCore::GraphicsLayer&);

    Ref<AsyncPDFRenderer> asyncRenderer();
    RefPtr<AsyncPDFRenderer> asyncRendererIfExists() const;
    void clearAsyncRenderer();

    Ref<UnifiedPDFPlugin> m_plugin;
    RefPtr<AsyncPDFRenderer> m_asyncRenderer;
};

} // namespace WebKit

#endif // ENABLE(UNIFIED_PDF)
