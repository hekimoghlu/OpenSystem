/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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

#include "LayerTreeContext.h"
#include "RemoteLayerBackingStoreCollection.h"
#include "RemoteLayerTreeTransaction.h"
#include <WebCore/FloatSize.h>
#include <WebCore/FrameIdentifier.h>
#include <WebCore/GraphicsLayerFactory.h>
#include <WebCore/HTMLMediaElementIdentifier.h>
#include <WebCore/LayerPool.h>
#include <WebCore/PlatformCALayer.h>
#include <wtf/CheckedPtr.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebKit {

class GraphicsLayerCARemote;
class PlatformCALayerRemote;
class RemoteRenderingBackendProxy;
class WebFrame;
class WebPage;

// FIXME: This class doesn't do much now. Roll into RemoteLayerTreeDrawingArea?
class RemoteLayerTreeContext : public RefCountedAndCanMakeWeakPtr<RemoteLayerTreeContext>, public WebCore::GraphicsLayerFactory {
    WTF_MAKE_TZONE_ALLOCATED(RemoteLayerTreeContext);
public:
    static Ref<RemoteLayerTreeContext> create(WebPage& webpage)
    {
        return adoptRef(*new RemoteLayerTreeContext(webpage));
    }

    ~RemoteLayerTreeContext();

    void layerDidEnterContext(PlatformCALayerRemote&, WebCore::PlatformCALayer::LayerType);
#if HAVE(AVKIT)
    void layerDidEnterContext(PlatformCALayerRemote&, WebCore::PlatformCALayer::LayerType, WebCore::HTMLVideoElement&);
#endif
    void layerWillLeaveContext(PlatformCALayerRemote&);

    void graphicsLayerDidEnterContext(GraphicsLayerCARemote&);
    void graphicsLayerWillLeaveContext(GraphicsLayerCARemote&);

    WebCore::LayerPool& layerPool() { return m_layerPool; }

    float deviceScaleFactor() const;

    LayerHostingMode layerHostingMode() const;
    
    std::optional<WebCore::DestinationColorSpace> displayColorSpace() const;

    std::optional<DrawingAreaIdentifier> drawingAreaIdentifier() const;

    void buildTransaction(RemoteLayerTreeTransaction&, WebCore::PlatformCALayer& rootLayer, WebCore::FrameIdentifier);

    void layerPropertyChangedWhileBuildingTransaction(PlatformCALayerRemote&);

    // From the UI process
    void animationDidStart(WebCore::PlatformLayerIdentifier, const String& key, MonotonicTime startTime);
    void animationDidEnd(WebCore::PlatformLayerIdentifier, const String& key);

    void willStartAnimationOnLayer(PlatformCALayerRemote&);

    RemoteLayerBackingStoreCollection& backingStoreCollection() { return *m_backingStoreCollection; }
    
    void setNextRenderingUpdateRequiresSynchronousImageDecoding() { m_nextRenderingUpdateRequiresSynchronousImageDecoding = true; }
    bool nextRenderingUpdateRequiresSynchronousImageDecoding() const { return m_nextRenderingUpdateRequiresSynchronousImageDecoding; }

    void adoptLayersFromContext(RemoteLayerTreeContext&);

    RemoteRenderingBackendProxy& ensureRemoteRenderingBackendProxy();

    bool useDynamicContentScalingDisplayListsForDOMRendering() const { return m_useDynamicContentScalingDisplayListsForDOMRendering; }
    void setUseDynamicContentScalingDisplayListsForDOMRendering(bool useDynamicContentScalingDisplayLists) { m_useDynamicContentScalingDisplayListsForDOMRendering = useDynamicContentScalingDisplayLists; }

    void gpuProcessConnectionWasDestroyed();

#if PLATFORM(IOS_FAMILY)
    bool canShowWhileLocked() const;
#endif

    WebPage& webPage();
    Ref<WebPage> protectedWebPage();

private:
    explicit RemoteLayerTreeContext(WebPage&);

    // WebCore::GraphicsLayerFactory
    Ref<WebCore::GraphicsLayer> createGraphicsLayer(WebCore::GraphicsLayer::Type, WebCore::GraphicsLayerClient&) override;

    WeakRef<WebPage> m_webPage;

    HashMap<WebCore::PlatformLayerIdentifier, RemoteLayerTreeTransaction::LayerCreationProperties> m_createdLayers;
    Vector<WebCore::PlatformLayerIdentifier> m_destroyedLayers;

    HashMap<WebCore::PlatformLayerIdentifier, WeakPtr<PlatformCALayerRemote>> m_livePlatformLayers;
    HashMap<WebCore::PlatformLayerIdentifier, WeakPtr<PlatformCALayerRemote>> m_layersWithAnimations;
#if HAVE(AVKIT)
    HashMap<WebCore::PlatformLayerIdentifier, PlaybackSessionContextIdentifier> m_videoLayers;
#endif

    HashSet<WeakRef<GraphicsLayerCARemote>> m_liveGraphicsLayers;

    UniqueRef<RemoteLayerBackingStoreCollection> m_backingStoreCollection;

    WebCore::LayerPool m_layerPool;

    CheckedPtr<RemoteLayerTreeTransaction> m_currentTransaction;

    bool m_nextRenderingUpdateRequiresSynchronousImageDecoding { false };
    bool m_useDynamicContentScalingDisplayListsForDOMRendering { false };
};

} // namespace WebKit
