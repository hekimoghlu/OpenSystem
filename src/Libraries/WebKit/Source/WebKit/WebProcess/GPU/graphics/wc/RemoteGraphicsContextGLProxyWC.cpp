/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 29, 2024.
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
#include "config.h"
#include "RemoteGraphicsContextGLProxy.h"

#if ENABLE(GPU_PROCESS) && ENABLE(WEBGL) && USE(GRAPHICS_LAYER_WC)

#include "GPUConnectionToWebProcess.h"
#include "GPUProcessConnection.h"
#include "RemoteGraphicsContextGLMessages.h"
#include "RemoteRenderingBackendProxy.h"
#include "WCPlatformLayerGCGL.h"
#include "WebProcess.h"
#include <WebCore/GraphicsLayerContentsDisplayDelegate.h>
#include <WebCore/TextureMapperPlatformLayer.h>

namespace WebKit {

namespace {

class PlatformLayerDisplayDelegate final : public WebCore::GraphicsLayerContentsDisplayDelegate {
public:
    static Ref<PlatformLayerDisplayDelegate> create(PlatformLayerContainer&& platformLayer)
    {
        return adoptRef(*new PlatformLayerDisplayDelegate(WTFMove(platformLayer)));
    }

    PlatformLayer* platformLayer() const final
    {
        return m_platformLayer.get();
    }

private:
    PlatformLayerDisplayDelegate(PlatformLayerContainer&& platformLayer)
        : m_platformLayer(WTFMove(platformLayer))
    {
    }

    PlatformLayerContainer m_platformLayer;
};

class RemoteGraphicsContextGLProxyWC final : public RemoteGraphicsContextGLProxy {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteGraphicsContextGLProxyWC);
public:
    // RemoteGraphicsContextGLProxy overrides.
    void prepareForDisplay() final;
    RefPtr<WebCore::GraphicsLayerContentsDisplayDelegate> layerContentsDisplayDelegate() final { return m_layerContentsDisplayDelegate.ptr(); }
private:
    explicit RemoteGraphicsContextGLProxyWC(const WebCore::GraphicsContextGLAttributes& attributes)
        : RemoteGraphicsContextGLProxy(attributes)
        , m_layerContentsDisplayDelegate(PlatformLayerDisplayDelegate::create(makeUnique<WCPlatformLayerGCGL>()))
    {
    }

    Ref<PlatformLayerDisplayDelegate> m_layerContentsDisplayDelegate;
    friend class RemoteGraphicsContextGLProxy;
};

void RemoteGraphicsContextGLProxyWC::prepareForDisplay()
{
    if (isContextLost())
        return;
    auto sendResult = sendSync(Messages::RemoteGraphicsContextGL::PrepareForDisplay());
    if (!sendResult.succeeded()) {
        markContextLost();
        return;
    }
    auto& [contentBuffer] = sendResult.reply();
    if (contentBuffer)
        static_cast<WCPlatformLayerGCGL*>(m_layerContentsDisplayDelegate->platformLayer())->addContentBufferIdentifier(*contentBuffer);
}

}

Ref<RemoteGraphicsContextGLProxy> RemoteGraphicsContextGLProxy::platformCreate(const WebCore::GraphicsContextGLAttributes& attributes)
{
    return adoptRef(*new RemoteGraphicsContextGLProxyWC(attributes));
}

}

#endif
