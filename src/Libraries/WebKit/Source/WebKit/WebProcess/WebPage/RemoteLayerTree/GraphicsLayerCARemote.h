/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 16, 2023.
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

#include <WebCore/GraphicsLayerCA.h>
#include <WebCore/HTMLMediaElementIdentifier.h>
#include <WebCore/PlatformLayer.h>
#include <wtf/TZoneMalloc.h>

#if ENABLE(MODEL_PROCESS)
namespace WebCore {
class ModelContext;
}
#endif

namespace WebKit {

class RemoteLayerTreeContext;

class GraphicsLayerCARemote final : public WebCore::GraphicsLayerCA, public CanMakeWeakPtr<GraphicsLayerCARemote> {
    WTF_MAKE_TZONE_ALLOCATED(GraphicsLayerCARemote);
public:
    GraphicsLayerCARemote(Type layerType, WebCore::GraphicsLayerClient&, RemoteLayerTreeContext&);
    virtual ~GraphicsLayerCARemote();

    bool filtersCanBeComposited(const WebCore::FilterOperations& filters) override;

    void moveToContext(RemoteLayerTreeContext&);
    LayerMode layerMode() const final;
    
private:
    bool isGraphicsLayerCARemote() const override { return true; }

    Ref<WebCore::PlatformCALayer> createPlatformCALayer(WebCore::PlatformCALayer::LayerType, WebCore::PlatformCALayerClient* owner) override;
    Ref<WebCore::PlatformCALayer> createPlatformCALayer(PlatformLayer*, WebCore::PlatformCALayerClient* owner) override;
#if ENABLE(MODEL_PROCESS)
    Ref<WebCore::PlatformCALayer> createPlatformCALayer(Ref<WebCore::ModelContext>, WebCore::PlatformCALayerClient* owner) override;
#endif
#if ENABLE(MODEL_ELEMENT)
    Ref<WebCore::PlatformCALayer> createPlatformCALayer(Ref<WebCore::Model>, WebCore::PlatformCALayerClient* owner) override;
#endif
#if HAVE(AVKIT)
    Ref<WebCore::PlatformCALayer> createPlatformVideoLayer(WebCore::HTMLVideoElement&, WebCore::PlatformCALayerClient* owner) override;
#endif
    Ref<WebCore::PlatformCAAnimation> createPlatformCAAnimation(WebCore::PlatformCAAnimation::AnimationType, const String& keyPath) override;
    Ref<WebCore::PlatformCALayer> createPlatformCALayerHost(WebCore::LayerHostingContextIdentifier, WebCore::PlatformCALayerClient*) override;

    // PlatformCALayerRemote can't currently proxy directly composited image contents, so opt out of this optimization.
    bool shouldDirectlyCompositeImage(WebCore::Image*) const override { return false; }

    bool shouldDirectlyCompositeImageBuffer(WebCore::ImageBuffer*) const override;
    void setLayerContentsToImageBuffer(WebCore::PlatformCALayer*, WebCore::ImageBuffer*) override;

    WebCore::Color pageTiledBackingBorderColor() const override;

    RefPtr<WebCore::GraphicsLayerAsyncContentsDisplayDelegate> createAsyncContentsDisplayDelegate(WebCore::GraphicsLayerAsyncContentsDisplayDelegate*) final;

    WeakPtr<RemoteLayerTreeContext> m_context;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_GRAPHICSLAYER(WebKit::GraphicsLayerCARemote, isGraphicsLayerCARemote())
