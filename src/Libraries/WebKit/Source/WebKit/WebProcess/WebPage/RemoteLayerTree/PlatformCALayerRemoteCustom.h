/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 7, 2023.
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

#include "PlatformCALayerRemote.h"

namespace WebCore {
class PlatformCALayerClient;

#if ENABLE(MODEL_PROCESS)
class ModelContext;
#endif
}

namespace WebKit {

class LayerHostingContext;

// PlatformCALayerRemoteCustom is used for CALayers that live in the web process and are hosted into the UI process via remote context.
class PlatformCALayerRemoteCustom final : public PlatformCALayerRemote {
    friend class PlatformCALayerRemote;
public:
    static Ref<PlatformCALayerRemote> create(PlatformLayer *, WebCore::PlatformCALayerClient*, RemoteLayerTreeContext&);
#if ENABLE(MODEL_PROCESS)
    static Ref<PlatformCALayerRemote> create(Ref<WebCore::ModelContext>, WebCore::PlatformCALayerClient*, RemoteLayerTreeContext&);
#endif
#if HAVE(AVKIT)
    static Ref<PlatformCALayerRemote> create(WebCore::HTMLVideoElement&, WebCore::PlatformCALayerClient* owner, RemoteLayerTreeContext&);
#endif

    virtual ~PlatformCALayerRemoteCustom();

    PlatformLayer* platformLayer() const override { return m_platformLayer.get(); }

    uint32_t hostingContextID() override;

    void setNeedsDisplayInRect(const WebCore::FloatRect& dirtyRect) override;
    void setNeedsDisplay() override;

    bool hasVideo() const { return m_hasVideo; }

private:
    PlatformCALayerRemoteCustom(WebCore::PlatformCALayer::LayerType, PlatformLayer *, WebCore::PlatformCALayerClient* owner, RemoteLayerTreeContext&);
    PlatformCALayerRemoteCustom(WebCore::PlatformCALayer::LayerType, LayerHostingContextID, WebCore::PlatformCALayerClient* owner, RemoteLayerTreeContext&);
    PlatformCALayerRemoteCustom(WebCore::HTMLVideoElement&, WebCore::PlatformCALayerClient* owner, RemoteLayerTreeContext&);
#if ENABLE(MODEL_PROCESS)
    PlatformCALayerRemoteCustom(WebCore::PlatformCALayer::LayerType, Ref<WebCore::ModelContext>, WebCore::PlatformCALayerClient* owner, RemoteLayerTreeContext&);
#endif

    Ref<WebCore::PlatformCALayer> clone(WebCore::PlatformCALayerClient* owner) const override;
    
    void populateCreationProperties(RemoteLayerTreeTransaction::LayerCreationProperties&, const RemoteLayerTreeContext&, WebCore::PlatformCALayer::LayerType) override;

    Type type() const final { return Type::RemoteCustom; }

    CFTypeRef contents() const override;
    void setContents(CFTypeRef) override;

    bool m_hasVideo { false };
    std::unique_ptr<LayerHostingContext> m_layerHostingContext;
    RetainPtr<PlatformLayer> m_platformLayer;
#if ENABLE(MODEL_PROCESS)
    RefPtr<WebCore::ModelContext> m_modelContext;
#endif
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_PLATFORM_CALAYER(WebKit::PlatformCALayerRemoteCustom, type() == WebCore::PlatformCALayer::Type::RemoteCustom)
