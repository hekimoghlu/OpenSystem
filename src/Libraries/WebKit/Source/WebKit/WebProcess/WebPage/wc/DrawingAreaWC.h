/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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

#if USE(GRAPHICS_LAYER_WC)

#include "DrawingArea.h"
#include "GraphicsLayerWC.h"
#include "RemoteWCLayerTreeHostProxy.h"
#include "WCLayerFactory.h"
#include <WebCore/GraphicsLayerFactory.h>
#include <WebCore/Timer.h>

namespace WebKit {

class DrawingAreaWC final
    : public DrawingArea
    , public GraphicsLayerWC::Observer {
public:
    static Ref<DrawingAreaWC> create(WebPage& webPage, const WebPageCreationParameters& parameters)
    {
        return adoptRef(*new DrawingAreaWC(webPage, parameters));
    }

    ~DrawingAreaWC() override;

private:
    DrawingAreaWC(WebPage&, const WebPageCreationParameters&);

    // DrawingArea
    WebCore::GraphicsLayerFactory* graphicsLayerFactory() override;
    void setNeedsDisplay() override;
    void setNeedsDisplayInRect(const WebCore::IntRect&) override;
    void scroll(const WebCore::IntRect& scrollRect, const WebCore::IntSize& scrollDelta) override;
    void updateRenderingWithForcedRepaintAsync(WebPage&, CompletionHandler<void()>&&) override;
    void triggerRenderingUpdate() override;
    bool enterAcceleratedCompositingModeIfNeeded() override { return false; }
    void setLayerTreeStateIsFrozen(bool) override;
    bool layerTreeStateIsFrozen() const override { return m_isRenderingSuspended; }
#if USE(GRAPHICS_LAYER_TEXTURE_MAPPER)    
    void updateGeometry(const WebCore::IntSize&, CompletionHandler<void()>&&) override { }
#endif
    void updateGeometryWC(uint64_t, WebCore::IntSize, float deviceScaleFactor, float intrinsicDeviceScaleFactor) override;
    void setRootCompositingLayer(WebCore::Frame&, WebCore::GraphicsLayer*) override;
    void addRootFrame(WebCore::FrameIdentifier) override;
    void attachViewOverlayGraphicsLayer(WebCore::FrameIdentifier, WebCore::GraphicsLayer*) override;
    void updatePreferences(const WebPreferencesStore&) override;
    bool shouldUseTiledBackingForFrameView(const WebCore::LocalFrameView&) const override;
    void displayDidRefresh() override;
    // GraphicsLayerWC::Observer
    void graphicsLayerAdded(GraphicsLayerWC&) override;
    void graphicsLayerRemoved(GraphicsLayerWC&) override;
    void commitLayerUpdateInfo(WCLayerUpdateInfo&&) override;
    RefPtr<WebCore::ImageBuffer> createImageBuffer(WebCore::FloatSize, float deviceScaleFactor) override;

    bool isCompositingMode();
    void updateRendering();
    void sendUpdateAC();
    void sendUpdateNonAC();
    void updateRootLayers();
    void updateRootLayerDeviceScaleFactor(WebCore::GraphicsLayer&);

    struct RootLayerInfo {
        Ref<WebCore::GraphicsLayer> layer;
        RefPtr<WebCore::GraphicsLayer> contentLayer;
        RefPtr<WebCore::GraphicsLayer> viewOverlayRootLayer;
        WebCore::FrameIdentifier frameID;
    };
    RootLayerInfo* rootLayerInfoWithFrameIdentifier(WebCore::FrameIdentifier);

    WebCore::GraphicsLayerClient m_rootLayerClient;
    std::unique_ptr<RemoteWCLayerTreeHostProxy> m_remoteWCLayerTreeHostProxy;
    WCLayerFactory m_layerFactory;
    DoublyLinkedList<GraphicsLayerWC> m_liveGraphicsLayers;
    WebCore::Timer m_updateRenderingTimer;
    bool m_isRenderingSuspended { false };
    bool m_hasDeferredRenderingUpdate { false };
    bool m_inUpdateRendering { false };
    bool m_waitDidUpdate { false };
    bool m_isForceRepaintCompletionHandlerDeferred { false };
    WCUpdateInfo m_updateInfo;
    Vector<RootLayerInfo, 1> m_rootLayers;
    Ref<WorkQueue> m_commitQueue;
    int64_t m_backingStoreStateID { 0 };
    WebCore::Region m_dirtyRegion;
    WebCore::IntRect m_scrollRect;
    WebCore::IntSize m_scrollOffset;
    CompletionHandler<void()> m_forceRepaintCompletionHandler;
};

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
