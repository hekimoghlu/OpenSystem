/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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

#if PLATFORM(MAC)

#include "CallbackID.h"
#include "DrawingArea.h"
#include "LayerTreeContext.h"
#include <WebCore/FloatRect.h>
#include <WebCore/TransformationMatrix.h>
#include <wtf/HashMap.h>
#include <wtf/RetainPtr.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS CALayer;

namespace WebCore {
class LocalFrameView;
class PlatformCALayer;
class RunLoopObserver;
class TiledBacking;
}

namespace WebKit {

class LayerHostingContext;

class TiledCoreAnimationDrawingArea final : public DrawingArea {
    WTF_MAKE_TZONE_ALLOCATED(TiledCoreAnimationDrawingArea);
public:
    static Ref<TiledCoreAnimationDrawingArea> create(WebPage& webPage, const WebPageCreationParameters& parameters)
    {
        return adoptRef(*new TiledCoreAnimationDrawingArea(webPage, parameters));
    }

    virtual ~TiledCoreAnimationDrawingArea();

private:
    TiledCoreAnimationDrawingArea(WebPage&, const WebPageCreationParameters&);

    // DrawingArea
    void setNeedsDisplay() override;
    void setNeedsDisplayInRect(const WebCore::IntRect&) override;
    void scroll(const WebCore::IntRect& scrollRect, const WebCore::IntSize& scrollDelta) override { }

    void updateRenderingWithForcedRepaint() override;
    void updateRenderingWithForcedRepaintAsync(WebPage&, CompletionHandler<void()>&&) override;
    void setLayerTreeStateIsFrozen(bool) override;
    bool layerTreeStateIsFrozen() const override;
    void setRootCompositingLayer(WebCore::Frame&, WebCore::GraphicsLayer*) override;
    void triggerRenderingUpdate() override;

    void updatePreferences(const WebPreferencesStore&) override;
    void mainFrameContentSizeChanged(WebCore::FrameIdentifier, const WebCore::IntSize&) override;

    void setViewExposedRect(std::optional<WebCore::FloatRect>) override;
    std::optional<WebCore::FloatRect> viewExposedRect() const override { return m_viewExposedRect; }

    WebCore::FloatRect exposedContentRect() const override;
    void setExposedContentRect(const WebCore::FloatRect&) override;

    bool supportsAsyncScrolling() const override { return true; }

    void registerScrollingTree() override;
    void unregisterScrollingTree() override;

    void dispatchAfterEnsuringUpdatedScrollPosition(WTF::Function<void ()>&&) override;

    bool shouldUseTiledBackingForFrameView(const WebCore::LocalFrameView&) const override;

    void activityStateDidChange(OptionSet<WebCore::ActivityState> changed, ActivityStateChangeID, CompletionHandler<void()>&&) override;

    void attachViewOverlayGraphicsLayer(WebCore::FrameIdentifier, WebCore::GraphicsLayer*) override;

    bool addMilestonesToDispatch(OptionSet<WebCore::LayoutMilestone> paintMilestones) override;

    void addCommitHandlers();

    enum class UpdateRenderingType { Normal, TransientZoom };
    void updateRendering(UpdateRenderingType = UpdateRenderingType::Normal);
    void didCompleteRenderingUpdateDisplay() override;

    // Message handlers.
    void updateGeometry(const WebCore::IntSize& viewSize, bool flushSynchronously, const WTF::MachSendRight& fencePort, CompletionHandler<void()>&&) override;
    void setDeviceScaleFactor(float, CompletionHandler<void()>&&) override;
    void suspendPainting();
    void resumePainting();
    void setLayerHostingMode(LayerHostingMode) override;
    void setColorSpace(std::optional<WebCore::DestinationColorSpace>) override;
    std::optional<WebCore::DestinationColorSpace> displayColorSpace() const override;
    void addFence(const WTF::MachSendRight&) override;

    void dispatchAfterEnsuringDrawing(IPC::AsyncReplyID) final;

    void sendEnterAcceleratedCompositingModeIfNeeded() override;
    void sendDidFirstLayerFlushIfNeeded();
    void handleActivityStateChangeCallbacksIfNeeded();
    void handleActivityStateChangeCallbacks();

    void adjustTransientZoom(double scale, WebCore::FloatPoint origin) override;
    void commitTransientZoom(double scale, WebCore::FloatPoint origin, CompletionHandler<void()>&&) override;
    void applyTransientZoomToPage(double scale, WebCore::FloatPoint origin);
    WebCore::PlatformCALayer* layerForTransientZoom() const;
    WebCore::PlatformCALayer* shadowLayerForTransientZoom() const;

    void applyTransientZoomToLayers(double scale, WebCore::FloatPoint origin);

    RefPtr<WebCore::DisplayRefreshMonitor> createDisplayRefreshMonitor(WebCore::PlatformDisplayID) final;

    void updateLayerHostingContext();

    void setRootCompositingLayer(CALayer *);
    void updateRootLayers();

    void updateDebugInfoLayer(bool showLayer);

    void sendPendingNewlyReachedPaintingMilestones();

    void scheduleRenderingUpdateRunLoopObserver();
    void invalidateRenderingUpdateRunLoopObserver();
    void renderingUpdateRunLoopCallback();

    void schedulePostRenderingUpdateRunLoopObserver();
    void invalidatePostRenderingUpdateRunLoopObserver();
    void postRenderingUpdateRunLoopCallback();

    void startRenderThrottlingTimer();
    void renderThrottlingTimerFired();

    std::unique_ptr<LayerHostingContext> m_layerHostingContext;

    RetainPtr<CALayer> m_hostingLayer;
    RetainPtr<CALayer> m_rootLayer;
    RetainPtr<CALayer> m_debugInfoLayer;
    RetainPtr<CALayer> m_pendingRootLayer;

    std::optional<WebCore::FloatRect> m_viewExposedRect;

    double m_transientZoomScale { 1 };
    WebCore::FloatPoint m_transientZoomOrigin;

    Vector<CompletionHandler<void()>> m_nextActivityStateChangeCallbacks;
    ActivityStateChangeID m_activityStateChangeID { ActivityStateChangeAsynchronous };

    RefPtr<WebCore::GraphicsLayer> m_viewOverlayRootLayer;

    OptionSet<WebCore::LayoutMilestone> m_pendingNewlyReachedPaintingMilestones;
    Vector<IPC::AsyncReplyID> m_pendingCallbackIDs;

    std::unique_ptr<WebCore::RunLoopObserver> m_renderingUpdateRunLoopObserver;
    std::unique_ptr<WebCore::RunLoopObserver> m_postRenderingUpdateRunLoopObserver;

    bool m_isPaintingSuspended { false };
    bool m_inUpdateGeometry { false };
    bool m_layerTreeStateIsFrozen { false };
    bool m_needsSendEnterAcceleratedCompositingMode { true };
    bool m_needsSendDidFirstLayerFlush { true };
    bool m_shouldHandleActivityStateChangeCallbacks { false };
    bool m_haveRegisteredHandlersForNextCommit { false };
};

inline bool TiledCoreAnimationDrawingArea::addMilestonesToDispatch(OptionSet<WebCore::LayoutMilestone> paintMilestones)
{
    m_pendingNewlyReachedPaintingMilestones.add(paintMilestones);
    return true;
}

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_DRAWING_AREA(TiledCoreAnimationDrawingArea, DrawingAreaType::TiledCoreAnimation)

#endif // PLATFORM(MAC)
