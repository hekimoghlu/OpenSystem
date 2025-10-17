/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 22, 2022.
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

#if USE(GRAPHICS_LAYER_TEXTURE_MAPPER)

#include "CallbackID.h"
#include "LayerTreeContext.h"
#include <WebCore/DisplayRefreshMonitor.h>
#include <WebCore/GLContext.h>
#include <WebCore/GraphicsLayerClient.h>
#include <WebCore/PlatformScreen.h>
#include <WebCore/TextureMapperFPSCounter.h>
#include <WebCore/Timer.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class GraphicsLayer;
class GraphicsLayerFactory;
class IntRect;
class IntSize;
class Page;
}

namespace WebKit {
class LayerTreeHost;
}

namespace WTF {
template<typename T> struct IsDeprecatedTimerSmartPointerException;
template<> struct IsDeprecatedTimerSmartPointerException<WebKit::LayerTreeHost> : std::true_type { };
}

namespace WebKit {

class WebPage;

class LayerTreeHost : public WebCore::GraphicsLayerClient {
    WTF_MAKE_TZONE_ALLOCATED(LayerTreeHost);
public:
    explicit LayerTreeHost(WebPage&);
    ~LayerTreeHost();

    const LayerTreeContext& layerTreeContext() const { return m_layerTreeContext; }
    void setLayerTreeStateIsFrozen(bool);
    void setShouldNotifyAfterNextScheduledLayerFlush(bool);
    void scheduleLayerFlush();
    void cancelPendingLayerFlush();
    void setRootCompositingLayer(WebCore::GraphicsLayer*);
    void setViewOverlayRootLayer(WebCore::GraphicsLayer*);
    void setNonCompositedContentsNeedDisplay(const WebCore::IntRect&);
    void scrollNonCompositedContents(const WebCore::IntRect&);
    void forceRepaint();
    void forceRepaintAsync(CompletionHandler<void()>&&);
    void sizeDidChange(const WebCore::IntSize& newSize);
    void pauseRendering();
    void resumeRendering();
    WebCore::GraphicsLayerFactory* graphicsLayerFactory();
    void contentsSizeChanged(const WebCore::IntSize&);
    void setIsDiscardable(bool);
    void backgroundColorDidChange();
    RefPtr<WebCore::DisplayRefreshMonitor> createDisplayRefreshMonitor(WebCore::PlatformDisplayID);
    WebCore::PlatformDisplayID displayID() const { return m_displayID; }

private:
    // GraphicsLayerClient
    void paintContents(const WebCore::GraphicsLayer*, WebCore::GraphicsContext&, const WebCore::FloatRect& rectToPaint, OptionSet<WebCore::GraphicsLayerPaintBehavior>) override;
    float deviceScaleFactor() const override;

    void initialize();
    GLNativeWindowType window();
    bool enabled();
    void compositeLayersToContext();
    void flushAndRenderLayers();
    bool flushPendingLayerChanges();
    void scrollNonCompositedContents(const WebCore::IntRect& scrollRect, const WebCore::IntSize& scrollOffset);
    void layerFlushTimerFired();
    bool prepareForRendering();
    void applyDeviceScaleFactor();

    WebPage& m_webPage;
    std::unique_ptr<WebCore::GLContext> m_context;
    LayerTreeContext m_layerTreeContext;
    WebCore::PlatformDisplayID m_displayID;
    RefPtr<WebCore::GraphicsLayer> m_rootLayer;
    WebCore::GraphicsLayer* m_rootCompositingLayer { nullptr };
    WebCore::GraphicsLayer* m_overlayCompositingLayer { nullptr };
    std::unique_ptr<WebCore::TextureMapper> m_textureMapper;
    WebCore::TextureMapperFPSCounter m_fpsCounter;
    WebCore::Timer m_layerFlushTimer;
    bool m_isSuspended { false };
};

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_TEXTURE_MAPPER)
