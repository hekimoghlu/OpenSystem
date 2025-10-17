/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#if USE(COORDINATED_GRAPHICS)
#include "CallbackID.h"
#include "LayerTreeContext.h"
#include "ThreadedCompositor.h"
#include <WebCore/CoordinatedImageBackingStore.h>
#include <WebCore/CoordinatedPlatformLayer.h>
#include <WebCore/FloatPoint.h>
#include <WebCore/GraphicsLayerClient.h>
#include <WebCore/GraphicsLayerFactory.h>
#include <WebCore/PlatformScreen.h>
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/OptionSet.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>

#if !HAVE(DISPLAY_LINK)
#include "ThreadedDisplayRefreshMonitor.h"
#endif

namespace WebCore {
class Damage;
class IntRect;
class IntSize;
class GraphicsLayer;
class GraphicsLayerFactory;
class NativeImage;
class SkiaPaintingEngine;
#if USE(CAIRO)
namespace Cairo {
class PaintingEngine;
}
#endif
}

namespace WebKit {
class LayerTreeHost;
}

namespace WTF {
template<typename T> struct IsDeprecatedTimerSmartPointerException;
template<> struct IsDeprecatedTimerSmartPointerException<WebKit::LayerTreeHost> : std::true_type { };
}

namespace WebKit {
class CoordinatedSceneState;
class WebPage;

class LayerTreeHost final : public CanMakeCheckedPtr<LayerTreeHost>, public WebCore::GraphicsLayerClient, public WebCore::GraphicsLayerFactory, public WebCore::CoordinatedPlatformLayer::Client
#if !HAVE(DISPLAY_LINK)
    , public ThreadedDisplayRefreshMonitor::Client
#endif
{
    WTF_MAKE_TZONE_ALLOCATED(LayerTreeHost);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LayerTreeHost);
public:
#if HAVE(DISPLAY_LINK)
    explicit LayerTreeHost(WebPage&);
#else
    LayerTreeHost(WebPage&, WebCore::PlatformDisplayID);
#endif
    ~LayerTreeHost();

    WebPage& webPage() const { return m_webPage; }
    CoordinatedSceneState& sceneState() const { return m_sceneState.get(); }

    const LayerTreeContext& layerTreeContext() const { return m_layerTreeContext; }
    void setLayerTreeStateIsFrozen(bool);

    void scheduleLayerFlush();
    void cancelPendingLayerFlush();
    void setRootCompositingLayer(WebCore::GraphicsLayer*);
    void setViewOverlayRootLayer(WebCore::GraphicsLayer*);

    void forceRepaint();
    void forceRepaintAsync(CompletionHandler<void()>&&);
    void sizeDidChange(const WebCore::IntSize&);

    void pauseRendering();
    void resumeRendering();

    WebCore::GraphicsLayerFactory* graphicsLayerFactory();

    void backgroundColorDidChange();

    void willRenderFrame();
    void didRenderFrame();
#if HAVE(DISPLAY_LINK)
    void didComposite(uint32_t);
#endif

#if !HAVE(DISPLAY_LINK)
    RefPtr<WebCore::DisplayRefreshMonitor> createDisplayRefreshMonitor(WebCore::PlatformDisplayID);
    WebCore::PlatformDisplayID displayID() const { return m_displayID; }
#endif

#if PLATFORM(GTK)
    void adjustTransientZoom(double, WebCore::FloatPoint);
    void commitTransientZoom(double, WebCore::FloatPoint);
#endif

#if PLATFORM(GTK) || PLATFORM(WPE)
    void ensureDrawing();
#endif

#if PLATFORM(WPE) && USE(GBM) && ENABLE(WPE_PLATFORM)
    void preferredBufferFormatsDidChange();
#endif
private:
    void updateRootLayer();
    WebCore::FloatRect visibleContentsRect() const;
    void layerFlushTimerFired();
    void flushLayers();
    void commitSceneState();
#if !HAVE(DISPLAY_LINK)
    void renderNextFrame(bool);
#endif

    // CoordinatedPlatformLayer::Client
#if USE(CAIRO)
    WebCore::Cairo::PaintingEngine& paintingEngine() override;
#elif USE(SKIA)
    WebCore::SkiaPaintingEngine& paintingEngine() const override { return *m_skiaPaintingEngine.get(); }
#endif
    Ref<WebCore::CoordinatedImageBackingStore> imageBackingStore(Ref<WebCore::NativeImage>&&) override;

    void attachLayer(WebCore::CoordinatedPlatformLayer&) override;
    void detachLayer(WebCore::CoordinatedPlatformLayer&) override;
    void notifyCompositionRequired() override;
    bool isCompositionRequiredOrOngoing() const override;
    void requestComposition() override;
    RunLoop* compositingRunLoop() const override;

    // GraphicsLayerFactory
    Ref<WebCore::GraphicsLayer> createGraphicsLayer(WebCore::GraphicsLayer::Type, WebCore::GraphicsLayerClient&) override;

#if !HAVE(DISPLAY_LINK)
    // ThreadedDisplayRefreshMonitor::Client
    void requestDisplayRefreshMonitorUpdate() override;
    void handleDisplayRefreshMonitorUpdate(bool hasBeenRescheduled) override;
#endif

#if PLATFORM(GTK)
    WebCore::FloatPoint constrainTransientZoomOrigin(double, WebCore::FloatPoint) const;
    WebCore::CoordinatedPlatformLayer* layerForTransientZoom() const;
    void applyTransientZoomToLayers(double, WebCore::FloatPoint);
#endif

    WebPage& m_webPage;
    LayerTreeContext m_layerTreeContext;
    Ref<CoordinatedSceneState> m_sceneState;
    WebCore::GraphicsLayer* m_rootCompositingLayer { nullptr };
    WebCore::GraphicsLayer* m_overlayCompositingLayer { nullptr };
    HashSet<Ref<WebCore::CoordinatedPlatformLayer>> m_layers;
    bool m_didInitializeRootCompositingLayer { false };
    bool m_layerTreeStateIsFrozen { false };
    bool m_isPurgingBackingStores { false };
    bool m_pendingResize { false };
    bool m_isSuspended { false };
    bool m_isWaitingForRenderer { false };
    bool m_scheduledWhileWaitingForRenderer { false };
    bool m_forceFrameSync { false };
    bool m_compositionRequired { false };
#if ENABLE(SCROLLING_THREAD)
    bool m_compositionRequiredInScrollingThread { false };
#endif
    double m_lastAnimationServiceTime { 0 };
    RefPtr<ThreadedCompositor> m_compositor;
    struct {
        CompletionHandler<void()> callback;
#if HAVE(DISPLAY_LINK)
        uint32_t compositionRequestID { 0 };
#else
        bool needsFreshFlush { false };
#endif
    } m_forceRepaintAsync;
    RunLoop::Timer m_layerFlushTimer;
#if !HAVE(DISPLAY_LINK)
    WebCore::PlatformDisplayID m_displayID;
#endif
#if USE(CAIRO)
    std::unique_ptr<WebCore::Cairo::PaintingEngine> m_paintingEngine;
#elif USE(SKIA)
    std::unique_ptr<WebCore::SkiaPaintingEngine> m_skiaPaintingEngine;
#endif
    HashMap<uint64_t, Ref<WebCore::CoordinatedImageBackingStore>> m_imageBackingStores;

#if PLATFORM(GTK)
    bool m_transientZoom { false };
    double m_transientZoomScale { 1 };
    WebCore::FloatPoint m_transientZoomOrigin;
#endif

    uint32_t m_compositionRequestID { 0 };
};

} // namespace WebKit

#endif // USE(COORDINATED_GRAPHICS)
