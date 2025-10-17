/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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

#include "PlaybackSessionContextIdentifier.h"
#include "RemoteLayerTreeNode.h"
#include "RemoteLayerTreeTransaction.h"
#include <WebCore/PlatformCALayer.h>
#include <WebCore/ProcessIdentifier.h>
#include <wtf/HashMap.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

#if PLATFORM(IOS_FAMILY) && ENABLE(MODEL_PROCESS)
#include <WebCore/ElementIdentifier.h>
#endif

OBJC_CLASS CAAnimation;
OBJC_CLASS WKAnimationDelegate;

namespace IPC {
class Connection;
}

namespace WebKit {

class RemoteLayerTreeDrawingAreaProxy;
class WebPageProxy;

class RemoteLayerTreeHost {
    WTF_MAKE_TZONE_ALLOCATED(RemoteLayerTreeHost);
public:
    explicit RemoteLayerTreeHost(RemoteLayerTreeDrawingAreaProxy&);
    ~RemoteLayerTreeHost();

    RemoteLayerTreeNode* nodeForID(std::optional<WebCore::PlatformLayerIdentifier>) const;
    RemoteLayerTreeNode* rootNode() const { return m_rootNode.get(); }
    RefPtr<RemoteLayerTreeNode> protectedRootNode() const { return m_rootNode.get(); }

    CALayer *layerForID(std::optional<WebCore::PlatformLayerIdentifier>) const;
    CALayer *rootLayer() const;

    RemoteLayerTreeDrawingAreaProxy& drawingArea() const;

    // Returns true if the root layer changed.
    bool updateLayerTree(const IPC::Connection&, const RemoteLayerTreeTransaction&, float indicatorScaleFactor  = 1);
    void asyncSetLayerContents(WebCore::PlatformLayerIdentifier, ImageBufferBackendHandle&&, const WebCore::RenderingResourceIdentifier&);

    void setIsDebugLayerTreeHost(bool flag) { m_isDebugLayerTreeHost = flag; }
    bool isDebugLayerTreeHost() const { return m_isDebugLayerTreeHost; }

    typedef HashMap<WebCore::PlatformLayerIdentifier, RetainPtr<WKAnimationDelegate>> LayerAnimationDelegateMap;
    LayerAnimationDelegateMap& animationDelegates() { return m_animationDelegates; }

    void animationDidStart(std::optional<WebCore::PlatformLayerIdentifier>, CAAnimation *, MonotonicTime startTime);
    void animationDidEnd(std::optional<WebCore::PlatformLayerIdentifier>, CAAnimation *);

#if ENABLE(THREADED_ANIMATION_RESOLUTION)
    void animationsWereAddedToNode(RemoteLayerTreeNode&);
    void animationsWereRemovedFromNode(RemoteLayerTreeNode&);
#endif

    void detachFromDrawingArea();
    void clearLayers();

    // Detach the root layer; it will be reattached upon the next incoming commit.
    void detachRootLayer();

    // Turn all CAMachPort objects in layer contents into actual IOSurfaces.
    // This avoids keeping an outstanding InUse reference when suspended.
    void mapAllIOSurfaceBackingStore();

    CALayer *layerWithIDForTesting(WebCore::PlatformLayerIdentifier) const;

    bool replayDynamicContentScalingDisplayListsIntoBackingStore() const;
    bool threadedAnimationResolutionEnabled() const;

    bool cssUnprefixedBackdropFilterEnabled() const;

#if ENABLE(THREADED_ANIMATION_RESOLUTION)
    Seconds acceleratedTimelineTimeOrigin(WebCore::ProcessIdentifier) const;
    MonotonicTime animationCurrentTime(WebCore::ProcessIdentifier) const;
#endif

    void remotePageProcessDidTerminate(WebCore::ProcessIdentifier);

private:
    void createLayer(const RemoteLayerTreeTransaction::LayerCreationProperties&);
    RefPtr<RemoteLayerTreeNode> makeNode(const RemoteLayerTreeTransaction::LayerCreationProperties&);

    bool updateBannerLayers(const RemoteLayerTreeTransaction&);

    void layerWillBeRemoved(WebCore::ProcessIdentifier, WebCore::PlatformLayerIdentifier);

    LayerContentsType layerContentsType() const;

    WeakPtr<RemoteLayerTreeDrawingAreaProxy> m_drawingArea;
    WeakPtr<RemoteLayerTreeNode> m_rootNode;
    HashMap<WebCore::PlatformLayerIdentifier, Ref<RemoteLayerTreeNode>> m_nodes;
    HashMap<WebCore::LayerHostingContextIdentifier, WebCore::PlatformLayerIdentifier> m_hostingLayers;
    HashMap<WebCore::LayerHostingContextIdentifier, WebCore::PlatformLayerIdentifier> m_hostedLayers;
    HashMap<WebCore::ProcessIdentifier, HashSet<WebCore::PlatformLayerIdentifier>> m_hostedLayersInProcess;
    HashMap<WebCore::PlatformLayerIdentifier, RetainPtr<WKAnimationDelegate>> m_animationDelegates;
#if HAVE(AVKIT)
    HashMap<WebCore::PlatformLayerIdentifier, PlaybackSessionContextIdentifier> m_videoLayers;
#endif
#if ENABLE(OVERLAY_REGIONS_IN_EVENT_REGION)
    HashSet<WebCore::PlatformLayerIdentifier> m_overlayRegionIDs;
#endif
#if PLATFORM(IOS_FAMILY) && ENABLE(MODEL_PROCESS)
    HashSet<WebCore::PlatformLayerIdentifier> m_modelLayers;
#endif
    bool m_isDebugLayerTreeHost { false };
};

} // namespace WebKit
