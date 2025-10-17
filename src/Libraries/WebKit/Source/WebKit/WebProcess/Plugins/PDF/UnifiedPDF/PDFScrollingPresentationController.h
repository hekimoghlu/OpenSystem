/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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

#include "PDFPresentationController.h"
#include <WebCore/GraphicsLayerClient.h>
#include <wtf/CheckedPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class KeyboardScrollingAnimator;
};

namespace WebKit {

class PDFScrollingPresentationController final : public PDFPresentationController, public WebCore::GraphicsLayerClient {
    WTF_MAKE_TZONE_ALLOCATED(PDFScrollingPresentationController);
    WTF_MAKE_NONCOPYABLE(PDFScrollingPresentationController);
public:
    PDFScrollingPresentationController(UnifiedPDFPlugin&);


private:
    bool supportsDisplayMode(PDFDocumentLayout::DisplayMode) const override;
    void willChangeDisplayMode(PDFDocumentLayout::DisplayMode) override { }

    void teardown() override;

    PDFPageCoverage pageCoverageForContentsRect(const WebCore::FloatRect&, std::optional<PDFLayoutRow>) const override;
    PDFPageCoverageAndScales pageCoverageAndScalesForContentsRect(const WebCore::FloatRect&, std::optional<PDFLayoutRow>, float tilingScaleFactor) const override;

    WebCore::FloatRect convertFromContentsToPainting(const WebCore::FloatRect& rect, std::optional<PDFDocumentLayout::PageIndex>) const override { return rect; }
    WebCore::FloatRect convertFromPaintingToContents(const WebCore::FloatRect& rect, std::optional<PDFDocumentLayout::PageIndex>) const override { return rect; }

    void deviceOrPageScaleFactorChanged() override { }

    void setupLayers(WebCore::GraphicsLayer& scrolledContentsLayer) override;
    void updateLayersOnLayoutChange(WebCore::FloatSize documentSize, WebCore::FloatSize centeringOffset, double scaleFactor) override;

    void updateIsInWindow(bool isInWindow) override;
    void updateDebugBorders(bool showDebugBorders, bool showRepaintCounters) override;
    void updateForCurrentScrollability(OptionSet<WebCore::TiledBackingScrollability>) override;

    GraphicsLayerClient& graphicsLayerClient() override { return *this; }

    bool handleKeyboardEvent(const WebKeyboardEvent&) override;
#if PLATFORM(MAC)
    bool handleKeyboardCommand(const WebKeyboardEvent&);
    CheckedPtr<WebCore::KeyboardScrollingAnimator> checkedKeyboardScrollingAnimator() const;
#endif

    std::optional<PDFDocumentLayout::PageIndex> pageIndexForCurrentView(AnchorPoint) const override;
    void restorePDFPosition(const VisiblePDFPosition&) override;

    void ensurePageIsVisible(PDFDocumentLayout::PageIndex) override { }

    // GraphicsLayerClient
    void notifyFlushRequired(const WebCore::GraphicsLayer*) override;
    float pageScaleFactor() const override;
    float deviceScaleFactor() const override;
    std::optional<float> customContentsScale(const WebCore::GraphicsLayer*) const override;
    void tiledBackingUsageChanged(const WebCore::GraphicsLayer*, bool /*usingTiledBacking*/) override;
    void paintContents(const WebCore::GraphicsLayer*, WebCore::GraphicsContext&, const WebCore::FloatRect&, OptionSet<WebCore::GraphicsLayerPaintBehavior>) override;

    void paintPDFSelection(const WebCore::GraphicsLayer*, WebCore::GraphicsContext&, const WebCore::FloatRect& clipRect, std::optional<PDFLayoutRow> = { });

    std::optional<WebCore::PlatformLayerIdentifier> contentsLayerIdentifier() const final;

    void updatePageBackgroundLayers();
    std::optional<PDFDocumentLayout::PageIndex> pageIndexForPageBackgroundLayer(const WebCore::GraphicsLayer*) const;
    WebCore::GraphicsLayer* backgroundLayerForPage(PDFDocumentLayout::PageIndex) const;

    void didGeneratePreviewForPage(PDFDocumentLayout::PageIndex) override;

    void paintBackgroundLayerForPage(const WebCore::GraphicsLayer*, WebCore::GraphicsContext&, const WebCore::FloatRect& clipRect, PDFDocumentLayout::PageIndex);

    Vector<LayerCoverage> layerCoveragesForRepaintPageCoverage(RepaintRequirements, const PDFPageCoverage&) override;

    RefPtr<WebCore::GraphicsLayer> m_pageBackgroundsContainerLayer;
    RefPtr<WebCore::GraphicsLayer> m_contentsLayer;
    RefPtr<WebCore::GraphicsLayer> m_selectionLayer;

    HashMap<RefPtr<WebCore::GraphicsLayer>, PDFDocumentLayout::PageIndex> m_pageBackgroundLayers;
};


} // namespace WebKit

#endif // ENABLE(UNIFIED_PDF)
