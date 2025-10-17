/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class PlatformWheelEvent;
}

namespace WebKit {

class UnifiedPDFPlugin;

enum class PageTransitionState : uint8_t {
    Idle,
    DeterminingStretchAxis,
    Stretching,
    Settling,
    StartingAnimationFromStationary,
    StartingAnimationFromMomentum,
    Animating
};

class PDFDiscretePresentationController final : public PDFPresentationController, public WebCore::GraphicsLayerClient {
    WTF_MAKE_TZONE_ALLOCATED(PDFDiscretePresentationController);
    WTF_MAKE_NONCOPYABLE(PDFDiscretePresentationController);
public:
    PDFDiscretePresentationController(UnifiedPDFPlugin&);


private:
    bool supportsDisplayMode(PDFDocumentLayout::DisplayMode) const override;
    void willChangeDisplayMode(PDFDocumentLayout::DisplayMode) override;

    void teardown() override;

    PDFPageCoverage pageCoverageForContentsRect(const WebCore::FloatRect&, std::optional<PDFLayoutRow>) const override;
    PDFPageCoverageAndScales pageCoverageAndScalesForContentsRect(const WebCore::FloatRect&, std::optional<PDFLayoutRow>, float tilingScaleFactor) const override;

    WebCore::FloatRect convertFromContentsToPainting(const WebCore::FloatRect&, std::optional<PDFDocumentLayout::PageIndex>) const override;
    WebCore::FloatRect convertFromPaintingToContents(const WebCore::FloatRect&, std::optional<PDFDocumentLayout::PageIndex>) const override;

    void deviceOrPageScaleFactorChanged() override;

    void setupLayers(WebCore::GraphicsLayer& scrolledContentsLayer) override;
    void updateLayersOnLayoutChange(WebCore::FloatSize documentSize, WebCore::FloatSize centeringOffset, double scaleFactor) override;

    void updateIsInWindow(bool isInWindow) override;
    void updateDebugBorders(bool showDebugBorders, bool showRepaintCounters) override;
    void updateForCurrentScrollability(OptionSet<TiledBackingScrollability>) override;

    Vector<LayerCoverage> layerCoveragesForRepaintPageCoverage(RepaintRequirements, const PDFPageCoverage&) override;

    void didGeneratePreviewForPage(PDFDocumentLayout::PageIndex) override;

    WebCore::GraphicsLayerClient& graphicsLayerClient() override { return *this; }

    std::optional<PDFLayoutRow> visibleRow() const override;
    std::optional<PDFLayoutRow> rowForLayer(const WebCore::GraphicsLayer*) const override;

    WebCore::FloatSize contentsOffsetForPage(PDFDocumentLayout::PageIndex) const;

    bool handleKeyboardEvent(const WebKeyboardEvent&) override;

    bool handleKeyboardCommand(const WebKeyboardEvent&);
    bool handleKeyboardEventForPageNavigation(const WebKeyboardEvent&);

    bool wantsWheelEvents() const override { return true; }
    bool handleWheelEvent(const WebWheelEvent&) override;

    bool handleBeginEvent(const WebCore::PlatformWheelEvent&);
    bool handleChangedEvent(const WebCore::PlatformWheelEvent&);
    bool handleEndedEvent(const WebCore::PlatformWheelEvent&);
    bool handleCancelledEvent(const WebCore::PlatformWheelEvent&);

    bool handleDiscreteWheelEvent(const WebCore::PlatformWheelEvent&);

    bool eventCanStartPageTransition(const PlatformWheelEvent&) const;

    bool canTransitionOnSide(WebCore::BoxSide) const;

    // Transition state
    enum class TransitionDirection : uint8_t {
        PreviousHorizontal,
        PreviousVertical,
        NextHorizontal,
        NextVertical
    };

    static constexpr bool isPreviousDirection(TransitionDirection direction)
    {
        return direction == TransitionDirection::PreviousHorizontal || direction == TransitionDirection::PreviousVertical;
    }

    static constexpr bool isNextDirection(TransitionDirection direction)
    {
        return direction == TransitionDirection::NextHorizontal || direction == TransitionDirection::NextVertical;
    }

    bool canTransitionInDirection(TransitionDirection) const;

    void startOrStopAnimationTimerIfNecessary();
    void animationTimerFired();
    void animateRubberBand(MonotonicTime);

    void startTransitionAnimation(PageTransitionState);
    void transitionEndTimerFired();

    void maybeEndGesture();
    void gestureEndTimerFired();
    void startPageTransitionOrSettle();

    void updateState(PageTransitionState);
    void updateLayerVisibilityForTransitionState(PageTransitionState previousState);
    void updateLayersForTransitionState();

    void applyWheelEventDelta(FloatSize);

    std::optional<PDFDocumentLayout::PageIndex> pageIndexForCurrentView(AnchorPoint) const override;
    void restorePDFPosition(const VisiblePDFPosition&) override;

    void ensurePageIsVisible(PDFDocumentLayout::PageIndex) override;

    // GraphicsLayerClient
    void notifyFlushRequired(const WebCore::GraphicsLayer*) override;
    float pageScaleFactor() const override;
    float deviceScaleFactor() const override;
    std::optional<float> customContentsScale(const WebCore::GraphicsLayer*) const override;
    void tiledBackingUsageChanged(const WebCore::GraphicsLayer*, bool /*usingTiledBacking*/) override;
    void paintContents(const WebCore::GraphicsLayer*, WebCore::GraphicsContext&, const WebCore::FloatRect&, OptionSet<WebCore::GraphicsLayerPaintBehavior>) override;

    void paintPDFSelection(const WebCore::GraphicsLayer*, WebCore::GraphicsContext&, const WebCore::FloatRect& clipRect, std::optional<PDFLayoutRow> = { });

    void paintBackgroundLayerForRow(const WebCore::GraphicsLayer*, WebCore::GraphicsContext&, const WebCore::FloatRect& clipRect, unsigned rowIndex);

    void buildRows();

    bool canGoToNextRow() const;
    bool canGoToPreviousRow() const;

    enum class Animated : bool { No, Yes };
    void goToNextRow(Animated);
    void goToPreviousRow(Animated);
    void goToRowIndex(unsigned rowIndex, Animated);

    void setVisibleRow(unsigned);
    void updateLayersAfterChangeInVisibleRow(std::optional<unsigned> additionalVisibleRowIndex = { });

    std::optional<unsigned> additionalVisibleRowIndexForDirection(TransitionDirection) const;

    // First index is layer, second index is start/end.
    static constexpr size_t topLayerIndex = 0;
    static constexpr size_t bottomLayerIndex = 1;
    static constexpr size_t startIndex = 0;
    static constexpr size_t endIndex = 1;
    std::array<std::array<float, 2>, 2> layerOpacitiesForStretchOffset(TransitionDirection, WebCore::FloatSize layerOffset, WebCore::FloatSize rowSize) const;
    WebCore::FloatSize layerOffsetForStretch(TransitionDirection, WebCore::FloatSize stretchDistance, WebCore::FloatSize rowSize) const;
    static float relevantAxisForDirection(TransitionDirection, WebCore::FloatSize);
    static void setRelevantAxisForDirection(TransitionDirection, WebCore::FloatSize&, float value);

    struct RowData {
        PDFLayoutRow pages;
        WebCore::FloatSize contentsOffset;
        RefPtr<WebCore::GraphicsLayer> containerLayer;
        RefPtr<WebCore::GraphicsLayer> leftPageContainerLayer;
        RefPtr<WebCore::GraphicsLayer> rightPageContainerLayer;
        RefPtr<WebCore::GraphicsLayer> contentsLayer;
        RefPtr<WebCore::GraphicsLayer> selectionLayer;

        bool isPageBackgroundLayer(const GraphicsLayer*) const;

        RefPtr<WebCore::GraphicsLayer> leftPageBackgroundLayer() const;
        RefPtr<WebCore::GraphicsLayer> rightPageBackgroundLayer() const;

        RefPtr<WebCore::GraphicsLayer> backgroundLayerForPageIndex(PDFDocumentLayout::PageIndex) const;

        RefPtr<WebCore::GraphicsLayer> protectedContainerLayer() const { return containerLayer; }
        RefPtr<WebCore::GraphicsLayer> protectedLeftPageContainerLayer() const { return leftPageContainerLayer; }
        RefPtr<WebCore::GraphicsLayer> protectedRightPageContainerLayer() const { return rightPageContainerLayer; }
        RefPtr<WebCore::GraphicsLayer> protectedContentsLayer() const { return contentsLayer; }
        RefPtr<WebCore::GraphicsLayer> protectedSelectionLayer() const { return selectionLayer; }
    };

    const RowData* rowDataForLayer(const WebCore::GraphicsLayer*) const;
    WebCore::FloatPoint positionForRowContainerLayer(const PDFLayoutRow&) const;
    WebCore::FloatSize rowContainerSize(const PDFLayoutRow&) const;

    RefPtr<WebCore::GraphicsLayer> protectedRowsContainerLayer() const { return m_rowsContainerLayer; }

    RefPtr<WebCore::GraphicsLayer> m_rowsContainerLayer;
    Vector<RowData> m_rows;

    HashMap<const WebCore::GraphicsLayer*, unsigned> m_layerToRowIndexMap;
    std::optional<PDFDocumentLayout::DisplayMode> m_displayModeAtLastLayerSetup;

    unsigned m_visibleRowIndex { 0 };

    // Gesture state
    WebCore::Timer m_gestureEndTimer { *this, &PDFDiscretePresentationController::gestureEndTimerFired };
    WebCore::Timer m_animationTimer { *this, &PDFDiscretePresentationController::animationTimerFired };
    WebCore::Timer m_transitionEndTimer { *this, &PDFDiscretePresentationController::transitionEndTimerFired };
    MonotonicTime m_animationStartTime;
    WebCore::FloatSize m_unappliedStretchDelta;
    WebCore::FloatSize m_stretchDistance;
    std::optional<WebCore::FloatSize> m_momentumVelocity;
    float m_animationStartDistance { 0 };
    PageTransitionState m_transitionState { PageTransitionState::Idle };
    std::optional<TransitionDirection> m_transitionDirection;
};


} // namespace WebKit

#endif // ENABLE(UNIFIED_PDF)
