/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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

#include "DrawingAreaProxy.h"
#include "LayerTreeContext.h"
#include <wtf/RefCounted.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>

#if !PLATFORM(WPE)
#include "BackingStore.h"
#endif

namespace WebCore {
class Region;
}

namespace WebKit {

class DrawingAreaProxyCoordinatedGraphics final : public DrawingAreaProxy, public RefCounted<DrawingAreaProxyCoordinatedGraphics> {
    WTF_MAKE_TZONE_ALLOCATED(DrawingAreaProxyCoordinatedGraphics);
    WTF_MAKE_NONCOPYABLE(DrawingAreaProxyCoordinatedGraphics);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DrawingAreaProxyCoordinatedGraphics);
public:
    static Ref<DrawingAreaProxyCoordinatedGraphics> create(WebPageProxy&, WebProcessProxy&);
    virtual ~DrawingAreaProxyCoordinatedGraphics();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

#if !PLATFORM(WPE)
    void paint(PlatformPaintContextPtr, const WebCore::IntRect&, WebCore::Region& unpaintedRegion);
#endif

    bool isInAcceleratedCompositingMode() const { return !m_layerTreeContext.isEmpty(); }
    const LayerTreeContext& layerTreeContext() const { return m_layerTreeContext; }

    void dispatchAfterEnsuringDrawing(CompletionHandler<void()>&&);

private:
    DrawingAreaProxyCoordinatedGraphics(WebPageProxy&, WebProcessProxy&);

    // DrawingAreaProxy
    void sizeDidChange() override;
    void deviceScaleFactorDidChange(CompletionHandler<void()>&&) override;
    void setBackingStoreIsDiscardable(bool) override;

#if HAVE(DISPLAY_LINK)
    std::optional<WebCore::FramesPerSecond> displayNominalFramesPerSecond() override;
#endif

#if PLATFORM(GTK)
    void adjustTransientZoom(double scale, WebCore::FloatPoint origin) override;
    void commitTransientZoom(double scale, WebCore::FloatPoint origin) override;
#endif

    // IPC message handlers
    void update(uint64_t backingStoreStateID, UpdateInfo&&) override;
    void enterAcceleratedCompositingMode(uint64_t backingStoreStateID, const LayerTreeContext&) override;
    void exitAcceleratedCompositingMode(uint64_t backingStoreStateID, UpdateInfo&&) override;
    void updateAcceleratedCompositingMode(uint64_t backingStoreStateID, const LayerTreeContext&) override;
    void dispatchPresentationCallbacksAfterFlushingLayers(IPC::Connection&, Vector<IPC::AsyncReplyID>&&) override;

    bool shouldSendWheelEventsToEventDispatcher() const override { return true; }

    bool alwaysUseCompositing() const;
    void enterAcceleratedCompositingMode(const LayerTreeContext&);
    void exitAcceleratedCompositingMode();
    void updateAcceleratedCompositingMode(const LayerTreeContext&);

    void sendUpdateGeometry();
    void didUpdateGeometry();

#if !PLATFORM(WPE)
    bool forceUpdateIfNeeded();
    void incorporateUpdate(UpdateInfo&&);
    void discardBackingStoreSoon();
    void discardBackingStore();
#endif

    class DrawingMonitor {
        WTF_MAKE_TZONE_ALLOCATED(DrawingMonitor);
        WTF_MAKE_NONCOPYABLE(DrawingMonitor);
    public:
        DrawingMonitor(WebPageProxy&);
        ~DrawingMonitor();

        void start(CompletionHandler<void()>&&);

    private:
        void stop();

        CompletionHandler<void()> m_callback;
        RunLoop::Timer m_timer;
    };

    // The current layer tree context.
    LayerTreeContext m_layerTreeContext;

    // Whether we're waiting for a DidUpdateGeometry message from the web process.
    bool m_isWaitingForDidUpdateGeometry { false };

    // The last size we sent to the web process.
    WebCore::IntSize m_lastSentSize;


#if !PLATFORM(WPE)
    bool m_isBackingStoreDiscardable { true };
    bool m_inForceUpdate { false };
    std::unique_ptr<BackingStore> m_backingStore;
    RunLoop::Timer m_discardBackingStoreTimer;
#endif
    std::unique_ptr<DrawingMonitor> m_drawingMonitor;
};

} // namespace WebKit

namespace WTF {
template<typename T> struct IsDeprecatedTimerSmartPointerException;
template<> struct IsDeprecatedTimerSmartPointerException<WebKit::DrawingAreaProxyCoordinatedGraphics::DrawingMonitor> : std::true_type { };
}

SPECIALIZE_TYPE_TRAITS_DRAWING_AREA_PROXY(DrawingAreaProxyCoordinatedGraphics, DrawingAreaType::CoordinatedGraphics)
